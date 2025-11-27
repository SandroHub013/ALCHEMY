"""
Integrazione completa con Microsoft Agent Lightning.

Agent Lightning è il framework per allenare agenti AI con:
- Reinforcement Learning (GRPO, PPO)
- Automatic Prompt Optimization (APO)
- Supervised Fine-Tuning (SFT) avanzato
- Tracciamento span per debugging

Questo modulo fornisce:
- AgentLightningTrainer: Wrapper completo per training RL
- RewardFunctions: Funzioni di reward per coding, function calling, etc.
- SpanTracker: Tracciamento dettagliato delle generazioni

Riferimenti:
- GitHub: https://github.com/microsoft/agent-lightning
- Docs: https://microsoft.github.io/agent-lightning/
"""

from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import re

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer, PreTrainedModel

# Agent Lightning imports
try:
    import agentlightning as agl
    from agentlightning import Trainer, LightningStore
    from agentlightning.algorithms import GRPO, SFT, APO
    from agentlightning.tracing import Span, Tracer
    AGENT_LIGHTNING_AVAILABLE = True
except ImportError:
    AGENT_LIGHTNING_AVAILABLE = False
    agl = None
    Trainer = None
    LightningStore = None
    GRPO = None
    SFT = None
    APO = None

logger = logging.getLogger(__name__)


class TrainingAlgorithm(str, Enum):
    """Algoritmi di training supportati da Agent Lightning."""
    SFT = "sft"           # Supervised Fine-Tuning
    GRPO = "grpo"         # Group Relative Policy Optimization (RL)
    APO = "apo"           # Automatic Prompt Optimization


@dataclass
class AgentLightningConfig:
    """Configurazione per Agent Lightning."""
    
    # Algoritmo di training
    algorithm: TrainingAlgorithm = TrainingAlgorithm.SFT
    
    # Configurazione GRPO (Reinforcement Learning)
    grpo_config: Dict[str, Any] = field(default_factory=lambda: {
        "num_generations": 4,      # Generazioni per prompt
        "temperature": 0.7,        # Temperatura per sampling
        "top_p": 0.9,             # Nucleus sampling
        "max_new_tokens": 512,    # Max token generati
        "kl_coef": 0.1,           # Coefficiente KL divergence
        "gamma": 0.99,            # Discount factor
        "clip_range": 0.2,        # PPO clip range
    })
    
    # Configurazione APO
    apo_config: Dict[str, Any] = field(default_factory=lambda: {
        "num_prompt_candidates": 5,
        "eval_samples": 20,
        "optimize_system_prompt": True,
    })
    
    # LightningStore
    store_path: str = "./lightning_store"
    enable_tracing: bool = True
    
    # Reward function da usare
    reward_function: str = "combined"  # "coding", "function_calling", "combined"


# =============================================================================
# REWARD FUNCTIONS - Il cuore del training RL
# =============================================================================

class RewardFunction:
    """
    Funzioni di reward per valutare le generazioni del modello.
    
    Il reward guida l'algoritmo RL (GRPO) a migliorare il comportamento dell'agente.
    """
    
    @staticmethod
    def coding_reward(
        prompt: str,
        generation: str,
        reference: Optional[str] = None,
    ) -> float:
        """
        Calcola il reward per task di coding.
        
        Criteri:
        - Sintassi corretta (prova a parsare)
        - Presenza di docstring
        - Lunghezza appropriata
        - Match con reference (se disponibile)
        
        Args:
            prompt: Il prompt originale
            generation: La risposta generata
            reference: Risposta di riferimento (opzionale)
            
        Returns:
            Reward float tra -1.0 e 1.0
        """
        reward = 0.0
        
        # 1. Estrai codice dalla generazione
        code_blocks = re.findall(r'```(?:python)?\n?(.*?)```', generation, re.DOTALL)
        if not code_blocks:
            # Se non ci sono code blocks ma sembra codice
            if 'def ' in generation or 'class ' in generation:
                code = generation
            else:
                return -0.5  # Penalizza assenza di codice
        else:
            code = code_blocks[0]
        
        # 2. Verifica sintassi Python
        try:
            compile(code, '<string>', 'exec')
            reward += 0.3  # Sintassi corretta
        except SyntaxError:
            reward -= 0.3  # Sintassi errata
        
        # 3. Presenza di docstring
        if '"""' in code or "'''" in code:
            reward += 0.1
        
        # 4. Presenza di type hints
        if ': ' in code and '->' in code:
            reward += 0.1
        
        # 5. Lunghezza appropriata (non troppo corta, non troppo lunga)
        code_len = len(code.strip())
        if 50 < code_len < 2000:
            reward += 0.1
        elif code_len < 20:
            reward -= 0.2  # Troppo corto
        
        # 6. Confronto con reference (se disponibile)
        if reference:
            # Semplice overlap di parole chiave
            ref_keywords = set(re.findall(r'\b\w+\b', reference.lower()))
            gen_keywords = set(re.findall(r'\b\w+\b', code.lower()))
            overlap = len(ref_keywords & gen_keywords) / max(len(ref_keywords), 1)
            reward += 0.4 * overlap
        
        return max(-1.0, min(1.0, reward))
    
    @staticmethod
    def function_calling_reward(
        prompt: str,
        generation: str,
        available_tools: Optional[List[str]] = None,
        reference: Optional[str] = None,
    ) -> float:
        """
        Calcola il reward per task di function calling.
        
        Criteri:
        - Formato corretto della chiamata (JSON valido)
        - Tool esistente (se lista disponibile)
        - Argomenti validi
        - Match con reference
        
        Args:
            prompt: Il prompt con la richiesta
            generation: La risposta con function call
            available_tools: Lista di tool disponibili
            reference: Chiamata di riferimento
            
        Returns:
            Reward float tra -1.0 e 1.0
        """
        reward = 0.0
        
        # 1. Cerca pattern di function call
        # Supporta vari formati: <function_call>, {"name": ...}, tool_call, etc.
        fc_patterns = [
            r'<function_call>\s*(\{.*?\})\s*</function_call>',
            r'"function_call":\s*(\{.*?\})',
            r'```json\s*(\{.*?\})\s*```',
            r'\{"name":\s*"[^"]+",\s*"arguments":\s*\{.*?\}\}',
        ]
        
        function_call = None
        for pattern in fc_patterns:
            matches = re.findall(pattern, generation, re.DOTALL)
            if matches:
                function_call = matches[0]
                break
        
        if not function_call:
            # Nessuna function call trovata
            if "function" in prompt.lower() or "tool" in prompt.lower():
                return -0.5  # Doveva chiamare una funzione
            return 0.0  # Non necessaria
        
        # 2. Verifica JSON valido
        try:
            fc_data = json.loads(function_call) if isinstance(function_call, str) else function_call
            reward += 0.3  # JSON valido
        except json.JSONDecodeError:
            return -0.3  # JSON non valido
        
        # 3. Verifica struttura corretta
        if "name" in fc_data:
            reward += 0.1
            
            # 4. Verifica tool esistente
            if available_tools:
                if fc_data["name"] in available_tools:
                    reward += 0.2
                else:
                    reward -= 0.2  # Tool non esistente
        
        if "arguments" in fc_data:
            reward += 0.1
            
            # Verifica che arguments sia un dict
            if isinstance(fc_data["arguments"], dict):
                reward += 0.1
        
        # 5. Confronto con reference
        if reference:
            try:
                ref_data = json.loads(reference)
                if fc_data.get("name") == ref_data.get("name"):
                    reward += 0.2
                    # Controlla argomenti
                    if fc_data.get("arguments") == ref_data.get("arguments"):
                        reward += 0.2
            except (json.JSONDecodeError, AttributeError):
                pass
        
        return max(-1.0, min(1.0, reward))
    
    @staticmethod
    def chat_reward(
        prompt: str,
        generation: str,
        reference: Optional[str] = None,
    ) -> float:
        """
        Calcola il reward per task di chat/conversazione.
        
        Criteri:
        - Risposta pertinente (non vuota, non troppo corta)
        - Non ripetitiva
        - Coerente con il prompt
        - Fluenza
        
        Args:
            prompt: La domanda/istruzione
            generation: La risposta generata
            reference: Risposta di riferimento
            
        Returns:
            Reward float tra -1.0 e 1.0
        """
        reward = 0.0
        
        # 1. Lunghezza appropriata
        gen_len = len(generation.strip())
        if gen_len < 10:
            return -0.5  # Troppo corto
        elif gen_len > 50:
            reward += 0.1
        
        # 2. Non ripetitivo (penalizza ripetizioni)
        words = generation.lower().split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                reward -= 0.3  # Molto ripetitivo
            elif unique_ratio > 0.7:
                reward += 0.2
        
        # 3. Risponde alla domanda (overlap keywords)
        prompt_words = set(re.findall(r'\b\w+\b', prompt.lower()))
        gen_words = set(re.findall(r'\b\w+\b', generation.lower()))
        
        # Rimuovi stopwords comuni
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                    'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                    'il', 'la', 'lo', 'i', 'le', 'gli', 'un', 'una', 'di', 'da',
                    'in', 'su', 'per', 'con', 'che', 'e', 'è', 'sono', 'come'}
        
        prompt_words -= stopwords
        gen_words -= stopwords
        
        if prompt_words:
            relevance = len(prompt_words & gen_words) / len(prompt_words)
            reward += 0.3 * relevance
        
        # 4. Match con reference
        if reference:
            ref_words = set(re.findall(r'\b\w+\b', reference.lower())) - stopwords
            if ref_words:
                similarity = len(gen_words & ref_words) / max(len(ref_words), 1)
                reward += 0.4 * similarity
        
        return max(-1.0, min(1.0, reward))
    
    @staticmethod
    def combined_reward(
        prompt: str,
        generation: str,
        task_type: str = "auto",
        reference: Optional[str] = None,
        available_tools: Optional[List[str]] = None,
    ) -> float:
        """
        Reward combinato che seleziona automaticamente la funzione giusta.
        
        Args:
            prompt: Il prompt
            generation: La generazione
            task_type: "coding", "function_calling", "chat", o "auto"
            reference: Riferimento opzionale
            available_tools: Tool disponibili (per function calling)
            
        Returns:
            Reward combinato
        """
        # Auto-detect task type
        if task_type == "auto":
            prompt_lower = prompt.lower()
            if any(kw in prompt_lower for kw in ['function', 'tool', 'call', 'api']):
                task_type = "function_calling"
            elif any(kw in prompt_lower for kw in ['code', 'python', 'function', 'class', 'write a']):
                task_type = "coding"
            else:
                task_type = "chat"
        
        if task_type == "coding":
            return RewardFunction.coding_reward(prompt, generation, reference)
        elif task_type == "function_calling":
            return RewardFunction.function_calling_reward(
                prompt, generation, available_tools, reference
            )
        else:
            return RewardFunction.chat_reward(prompt, generation, reference)


# =============================================================================
# AGENT LIGHTNING TRAINER
# =============================================================================

class AgentLightningTrainer:
    """
    Trainer completo con Agent Lightning per training RL di agenti AI.
    
    Integra:
    - GRPO per Reinforcement Learning
    - APO per Automatic Prompt Optimization  
    - SFT per Supervised Fine-Tuning
    - Tracciamento span per debugging
    - LightningStore per gestione risorse
    
    Esempio:
        ```python
        trainer = AgentLightningTrainer(
            model=model,
            tokenizer=tokenizer,
            config=AgentLightningConfig(algorithm=TrainingAlgorithm.GRPO)
        )
        
        # Training con RL
        trainer.train(train_dataset, eval_dataset)
        
        # Genera con tracciamento
        output = trainer.generate("Write a Python function...")
        ```
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: AgentLightningConfig,
        reward_fn: Optional[Callable] = None,
    ):
        """
        Inizializza il trainer.
        
        Args:
            model: Modello con LoRA/PEFT applicato
            tokenizer: Tokenizer
            config: Configurazione Agent Lightning
            reward_fn: Funzione di reward custom (opzionale)
        """
        if not AGENT_LIGHTNING_AVAILABLE:
            raise ImportError(
                "Agent Lightning non installato. Installa con: pip install agentlightning"
            )
        
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Imposta reward function
        if reward_fn is not None:
            self.reward_fn = reward_fn
        else:
            self.reward_fn = self._get_default_reward_fn()
        
        # Inizializza LightningStore
        self.store = LightningStore(path=config.store_path)
        
        # Inizializza Tracer se abilitato
        self.tracer = Tracer() if config.enable_tracing else None
        
        # Inizializza algoritmo
        self.algorithm = self._create_algorithm()
        
        # Trainer Agent Lightning
        self.trainer = None
        
        logger.info(f"AgentLightningTrainer inizializzato con algoritmo: {config.algorithm.value}")
    
    def _get_default_reward_fn(self) -> Callable:
        """Ritorna la reward function di default basata sulla config."""
        reward_type = self.config.reward_function
        
        if reward_type == "coding":
            return lambda p, g, r=None: RewardFunction.coding_reward(p, g, r)
        elif reward_type == "function_calling":
            return lambda p, g, r=None: RewardFunction.function_calling_reward(p, g, reference=r)
        else:  # combined
            return lambda p, g, r=None: RewardFunction.combined_reward(p, g, reference=r)
    
    def _create_algorithm(self):
        """Crea l'algoritmo di training appropriato."""
        if self.config.algorithm == TrainingAlgorithm.GRPO:
            return GRPO(
                model=self.model,
                tokenizer=self.tokenizer,
                reward_fn=self.reward_fn,
                **self.config.grpo_config,
            )
        elif self.config.algorithm == TrainingAlgorithm.APO:
            return APO(
                model=self.model,
                tokenizer=self.tokenizer,
                **self.config.apo_config,
            )
        else:  # SFT
            return SFT(
                model=self.model,
                tokenizer=self.tokenizer,
            )
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        output_dir: str = "./checkpoints",
    ) -> Dict[str, Any]:
        """
        Avvia il training con Agent Lightning.
        
        Args:
            train_dataset: Dataset di training
            eval_dataset: Dataset di valutazione (opzionale)
            num_epochs: Numero di epoche
            batch_size: Batch size
            learning_rate: Learning rate
            output_dir: Directory per checkpoint
            
        Returns:
            Dizionario con metriche di training
        """
        logger.info(f"Avvio training con {self.config.algorithm.value}")
        logger.info(f"  - Epochs: {num_epochs}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Learning rate: {learning_rate}")
        
        # Crea Trainer Agent Lightning
        self.trainer = Trainer(
            algorithm=self.algorithm,
            store=self.store,
            output_dir=output_dir,
        )
        
        # Avvia training
        with self._trace_span("training", {"algorithm": self.config.algorithm.value}):
            results = self.trainer.train(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )
        
        # Salva modello finale
        self.save_model(output_dir)
        
        return results
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        trace: bool = True,
    ) -> str:
        """
        Genera una risposta con tracciamento opzionale.
        
        Args:
            prompt: Il prompt di input
            max_new_tokens: Numero massimo di token da generare
            temperature: Temperatura per sampling
            top_p: Nucleus sampling threshold
            trace: Se tracciare la generazione
            
        Returns:
            Testo generato
        """
        # Tokenizza input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Genera con tracciamento
        with self._trace_span("generation", {"prompt_len": len(prompt)}) as span:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            
            if span:
                span.set_attribute("output_len", len(generated_text))
        
        # Emetti evento per tracciamento
        if self.config.enable_tracing:
            agl.emit_generation(
                prompt=prompt,
                response=generated_text,
                model=self.model.config._name_or_path,
            )
        
        return generated_text
    
    def evaluate_reward(
        self,
        prompt: str,
        generation: str,
        reference: Optional[str] = None,
    ) -> float:
        """
        Valuta il reward per una generazione.
        
        Args:
            prompt: Prompt originale
            generation: Testo generato
            reference: Riferimento opzionale
            
        Returns:
            Valore di reward
        """
        return self.reward_fn(prompt, generation, reference)
    
    def save_model(self, output_dir: str) -> None:
        """Salva il modello e l'adattatore LoRA."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Salva adapter LoRA
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Salva config Agent Lightning
        config_path = os.path.join(output_dir, "agent_lightning_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "algorithm": self.config.algorithm.value,
                "grpo_config": self.config.grpo_config,
                "apo_config": self.config.apo_config,
                "reward_function": self.config.reward_function,
            }, f, indent=2)
        
        logger.info(f"Modello salvato in: {output_dir}")
    
    def _trace_span(self, name: str, attributes: Optional[Dict] = None):
        """Context manager per tracciamento span."""
        if self.tracer and self.config.enable_tracing:
            return self.tracer.span(name, attributes=attributes or {})
        
        # Dummy context manager se tracing disabilitato
        from contextlib import nullcontext
        return nullcontext()


# =============================================================================
# FACTORY E UTILITIES
# =============================================================================

def create_agent_lightning_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: Dict[str, Any],
) -> AgentLightningTrainer:
    """
    Factory function per creare AgentLightningTrainer dalla config YAML.
    
    Args:
        model: Modello
        tokenizer: Tokenizer
        config: Config completa (da config.yaml)
        
    Returns:
        AgentLightningTrainer configurato
    """
    agl_config = config.get("agent_lightning", {})
    
    algorithm_str = agl_config.get("algorithm", "sft")
    try:
        algorithm = TrainingAlgorithm(algorithm_str.lower())
    except ValueError:
        logger.warning(f"Algoritmo '{algorithm_str}' non valido, uso SFT")
        algorithm = TrainingAlgorithm.SFT
    
    trainer_config = AgentLightningConfig(
        algorithm=algorithm,
        grpo_config=agl_config.get("grpo", {}),
        apo_config=agl_config.get("apo", {}),
        store_path=agl_config.get("store_path", "./lightning_store"),
        enable_tracing=agl_config.get("enable_tracing", True),
        reward_function=agl_config.get("reward_function", "combined"),
    )
    
    return AgentLightningTrainer(
        model=model,
        tokenizer=tokenizer,
        config=trainer_config,
    )


def check_agent_lightning_available() -> bool:
    """Verifica se Agent Lightning è installato."""
    return AGENT_LIGHTNING_AVAILABLE

