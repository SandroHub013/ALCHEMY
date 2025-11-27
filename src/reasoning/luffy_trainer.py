"""
LUFFY: Learning to Reason under Off-Policy Guidance.

Implementazione ispirata al paper "Learning to Reason under Off-Policy Guidance"
(https://arxiv.org/abs/2504.14945)

LUFFY è un framework generale per l'apprendimento off-policy in Large Reasoning Models:
- Integra tracce di ragionamento off-policy (es. da DeepSeek-R1)
- Combina generazioni on-policy del modello con guidance esterna
- Migliora significativamente le capacità di ragionamento matematico e logico

Include ExGRPO: una variante che impara dall'esperienza off-policy del modello stesso.

Riferimenti:
- GitHub: https://github.com/ElliottYan/LUFFY
- Paper: https://arxiv.org/abs/2504.14945
- ExGRPO Paper: https://arxiv.org/abs/2510.02245
"""

from typing import Optional, Dict, Any, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)


class OffPolicyMode(str, Enum):
    """Modalità di apprendimento off-policy supportate."""
    LUFFY = "luffy"           # Off-policy guidance da modello esterno (es. DeepSeek-R1)
    EXGRPO = "exgrpo"         # Off-policy dall'esperienza del modello stesso
    HYBRID = "hybrid"         # Combinazione di entrambi


@dataclass
class LuffyConfig:
    """
    Configurazione per LUFFY trainer.
    
    Attributes:
        mode: Modalità off-policy (luffy, exgrpo, hybrid)
        off_policy_source: Fonte delle tracce off-policy (es. "deepseek-r1")
        off_policy_weight: Peso delle tracce off-policy nel training
        on_policy_weight: Peso delle generazioni on-policy
        temperature: Temperatura per sampling
        num_generations: Numero di risposte generate per prompt
        max_new_tokens: Massimo token generati
        kl_coef: Coefficiente KL divergence
        clip_range: PPO clip range
        value_coef: Coefficiente per value loss
        entropy_coef: Coefficiente per entropy bonus
    """
    
    mode: OffPolicyMode = OffPolicyMode.LUFFY
    
    # Off-policy configuration
    off_policy_source: str = "deepseek-r1"
    off_policy_weight: float = 0.5
    on_policy_weight: float = 0.5
    
    # Filtering thresholds
    min_off_policy_reward: float = 0.5  # Soglia minima reward per usare tracce off-policy
    
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.95
    num_generations: int = 4
    max_new_tokens: int = 2048  # Più lungo per ragionamento
    
    # RL parameters
    kl_coef: float = 0.05
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # ExGRPO specific
    experience_buffer_size: int = 10000
    experience_sample_ratio: float = 0.3  # Percentuale di esperienza da riutilizzare


@dataclass
class ExGRPOConfig:
    """
    Configurazione specifica per ExGRPO.
    
    ExGRPO impara dall'esperienza off-policy del modello stesso,
    senza bisogno di guidance esterna.
    """
    
    # Experience replay
    buffer_size: int = 10000
    sample_ratio: float = 0.3
    priority_alpha: float = 0.6  # Prioritized experience replay
    priority_beta: float = 0.4
    
    # Filtering
    min_reward_threshold: float = 0.6
    max_reward_threshold: float = 1.0
    
    # Mixing
    experience_weight: float = 0.3
    on_policy_weight: float = 0.7


class OffPolicyDataMixer:
    """
    Mixer per combinare dati on-policy e off-policy.
    
    Gestisce:
    - Caricamento tracce off-policy (es. da DeepSeek-R1)
    - Filtraggio basato su reward
    - Bilanciamento con generazioni on-policy
    - Experience buffer per ExGRPO
    """
    
    def __init__(
        self,
        config: LuffyConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        self.config = config
        self.tokenizer = tokenizer
        
        # Buffer per tracce off-policy
        self.off_policy_traces: List[Dict[str, Any]] = []
        
        # Experience buffer per ExGRPO
        self.experience_buffer: List[Dict[str, Any]] = []
        self.experience_priorities: List[float] = []
        
        logger.info(f"OffPolicyDataMixer inizializzato in modalità: {config.mode.value}")
    
    def load_off_policy_traces(
        self,
        traces_path: str,
        source_model: str = "deepseek-r1",
    ) -> int:
        """
        Carica tracce di ragionamento off-policy.
        
        Args:
            traces_path: Path al file JSON con le tracce
            source_model: Nome del modello sorgente
            
        Returns:
            Numero di tracce caricate
        """
        logger.info(f"Caricamento tracce off-policy da: {traces_path}")
        
        with open(traces_path, "r", encoding="utf-8") as f:
            traces = json.load(f)
        
        # Filtra per reward minimo
        filtered_traces = []
        for trace in traces:
            if trace.get("reward", 0) >= self.config.min_off_policy_reward:
                trace["source_model"] = source_model
                filtered_traces.append(trace)
        
        self.off_policy_traces.extend(filtered_traces)
        
        logger.info(
            f"Caricate {len(filtered_traces)}/{len(traces)} tracce "
            f"(filtrate per reward >= {self.config.min_off_policy_reward})"
        )
        
        return len(filtered_traces)
    
    def add_to_experience_buffer(
        self,
        prompt: str,
        response: str,
        reward: float,
        log_prob: float,
    ) -> None:
        """
        Aggiunge un'esperienza al buffer per ExGRPO.
        
        Args:
            prompt: Prompt di input
            response: Risposta generata
            reward: Reward ottenuto
            log_prob: Log probability della risposta
        """
        experience = {
            "prompt": prompt,
            "response": response,
            "reward": reward,
            "log_prob": log_prob,
            "timestamp": len(self.experience_buffer),
        }
        
        # Priority basata su reward (esperienze migliori più probabili)
        priority = (reward + 1.0) ** self.config.experience_sample_ratio
        
        # Gestisci buffer size
        if len(self.experience_buffer) >= self.config.experience_buffer_size:
            # Rimuovi esperienza con priorità più bassa
            min_idx = self.experience_priorities.index(min(self.experience_priorities))
            self.experience_buffer.pop(min_idx)
            self.experience_priorities.pop(min_idx)
        
        self.experience_buffer.append(experience)
        self.experience_priorities.append(priority)
    
    def sample_mixed_batch(
        self,
        on_policy_data: List[Dict[str, Any]],
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        """
        Campiona un batch misto on-policy/off-policy.
        
        Args:
            on_policy_data: Dati generati on-policy
            batch_size: Dimensione del batch
            
        Returns:
            Batch misto con flag per ogni elemento
        """
        mixed_batch = []
        
        # Calcola split
        off_policy_count = int(batch_size * self.config.off_policy_weight)
        on_policy_count = batch_size - off_policy_count
        
        # Campiona off-policy
        if self.config.mode == OffPolicyMode.EXGRPO:
            # Usa experience buffer
            off_policy_samples = self._sample_from_experience_buffer(off_policy_count)
        else:
            # Usa tracce esterne
            off_policy_samples = self._sample_off_policy_traces(off_policy_count)
        
        for sample in off_policy_samples:
            sample["is_off_policy"] = True
            mixed_batch.append(sample)
        
        # Campiona on-policy
        on_policy_samples = random.sample(
            on_policy_data, 
            min(on_policy_count, len(on_policy_data))
        )
        for sample in on_policy_samples:
            sample["is_off_policy"] = False
            mixed_batch.append(sample)
        
        # Shuffle
        random.shuffle(mixed_batch)
        
        return mixed_batch
    
    def _sample_off_policy_traces(self, count: int) -> List[Dict[str, Any]]:
        """Campiona tracce off-policy."""
        if not self.off_policy_traces:
            return []
        
        return random.sample(
            self.off_policy_traces,
            min(count, len(self.off_policy_traces))
        )
    
    def _sample_from_experience_buffer(self, count: int) -> List[Dict[str, Any]]:
        """Campiona dal buffer di esperienza con prioritized replay."""
        if not self.experience_buffer:
            return []
        
        # Normalizza priorità
        total_priority = sum(self.experience_priorities)
        probs = [p / total_priority for p in self.experience_priorities]
        
        # Campiona con priorità
        indices = random.choices(
            range(len(self.experience_buffer)),
            weights=probs,
            k=min(count, len(self.experience_buffer))
        )
        
        return [self.experience_buffer[i] for i in indices]


class LuffyTrainer:
    """
    Trainer principale per LUFFY.
    
    Implementa l'algoritmo LUFFY per apprendimento off-policy:
    1. Genera risposte on-policy
    2. Recupera/campiona tracce off-policy
    3. Combina con pesi configurabili
    4. Applica GRPO con importance sampling
    
    Include supporto per ExGRPO (apprendimento dall'esperienza propria).
    
    Esempio:
        ```python
        trainer = LuffyTrainer(
            model=model,
            tokenizer=tokenizer,
            config=LuffyConfig(mode=OffPolicyMode.LUFFY)
        )
        
        # Carica tracce da DeepSeek-R1
        trainer.load_off_policy_traces("deepseek_r1_traces.json")
        
        # Training
        trainer.train(train_dataset, num_epochs=3)
        ```
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: LuffyConfig,
        reward_fn: Optional[Callable] = None,
        reference_model: Optional[PreTrainedModel] = None,
    ):
        """
        Inizializza LUFFY Trainer.
        
        Args:
            model: Modello da addestrare
            tokenizer: Tokenizer
            config: Configurazione LUFFY
            reward_fn: Funzione di reward custom
            reference_model: Modello di riferimento per KL divergence
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn or self._default_reward
        self.reference_model = reference_model
        
        # Data mixer
        self.data_mixer = OffPolicyDataMixer(config, tokenizer)
        
        # Statistiche training
        self.training_stats = {
            "total_steps": 0,
            "on_policy_rewards": [],
            "off_policy_rewards": [],
            "kl_divergence": [],
            "policy_loss": [],
        }
        
        logger.info(f"LuffyTrainer inizializzato - Mode: {config.mode.value}")
        
        if config.mode == OffPolicyMode.EXGRPO:
            logger.info("ExGRPO mode: il modello imparerà dalla propria esperienza")
    
    def load_off_policy_traces(self, traces_path: str) -> int:
        """
        Carica tracce off-policy da file.
        
        Args:
            traces_path: Path al file JSON
            
        Returns:
            Numero di tracce caricate
        """
        return self.data_mixer.load_off_policy_traces(
            traces_path,
            source_model=self.config.off_policy_source
        )
    
    def _default_reward(self, prompt: str, response: str) -> float:
        """Reward function di default per math reasoning."""
        # Semplice euristica - in produzione usare Math-Verify
        reward = 0.0
        
        # Presenza di step di ragionamento
        if "let me" in response.lower() or "step" in response.lower():
            reward += 0.2
        
        # Presenza di formule/equazioni
        if "=" in response and any(c.isdigit() for c in response):
            reward += 0.2
        
        # Conclusione chiara
        if "therefore" in response.lower() or "answer" in response.lower():
            reward += 0.2
        
        # Lunghezza ragionevole (ragionamento strutturato)
        if 200 < len(response) < 3000:
            reward += 0.2
        
        # Box answer (formato comune per math)
        if "\\boxed" in response or "**answer**" in response.lower():
            reward += 0.2
        
        return min(1.0, reward)
    
    def generate_on_policy(
        self,
        prompts: List[str],
        num_generations: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Genera risposte on-policy.
        
        Args:
            prompts: Lista di prompt
            num_generations: Numero di generazioni per prompt
            
        Returns:
            Lista di dizionari con prompt, response, reward, log_prob
        """
        num_gens = num_generations or self.config.num_generations
        results = []
        
        self.model.eval()
        
        for prompt in prompts:
            # Tokenizza
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Genera multiple risposte
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    num_return_sequences=num_gens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Processa ogni generazione
            for i in range(num_gens):
                response = self.tokenizer.decode(
                    outputs.sequences[i][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                
                # Calcola reward
                reward = self.reward_fn(prompt, response)
                
                # Calcola log probability (semplificato)
                log_prob = self._compute_log_prob(prompt, response)
                
                result = {
                    "prompt": prompt,
                    "response": response,
                    "reward": reward,
                    "log_prob": log_prob,
                    "is_off_policy": False,
                }
                results.append(result)
                
                # Aggiungi a experience buffer per ExGRPO
                if self.config.mode in [OffPolicyMode.EXGRPO, OffPolicyMode.HYBRID]:
                    self.data_mixer.add_to_experience_buffer(
                        prompt, response, reward, log_prob
                    )
        
        return results
    
    def _compute_log_prob(self, prompt: str, response: str) -> float:
        """Calcola log probability della risposta."""
        full_text = prompt + response
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Shift per next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        
        # Log softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs per token
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # Media (per evitare bias per lunghezza)
        return token_log_probs.mean().item()
    
    def compute_grpo_loss(
        self,
        batch: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calcola GRPO loss con off-policy correction.
        
        Args:
            batch: Batch di dati (on-policy + off-policy)
            
        Returns:
            Loss e metriche
        """
        total_loss = 0.0
        metrics = {
            "policy_loss": 0.0,
            "kl_loss": 0.0,
            "on_policy_reward_mean": 0.0,
            "off_policy_reward_mean": 0.0,
        }
        
        on_policy_rewards = []
        off_policy_rewards = []
        
        for item in batch:
            prompt = item["prompt"]
            response = item["response"]
            reward = item["reward"]
            old_log_prob = item.get("log_prob", 0.0)
            is_off_policy = item.get("is_off_policy", False)
            
            # Calcola log prob attuale
            current_log_prob = self._compute_log_prob(prompt, response)
            
            # Importance sampling ratio
            ratio = torch.exp(torch.tensor(current_log_prob - old_log_prob))
            
            # Calcola advantage (normalizzato nel gruppo)
            advantage = reward  # Semplificato - usare GAE in produzione
            
            # Policy loss con clipping
            surr1 = ratio * advantage
            surr2 = torch.clamp(
                ratio, 
                1 - self.config.clip_range,
                1 + self.config.clip_range
            ) * advantage
            policy_loss = -torch.min(surr1, surr2)
            
            # Weight per off-policy
            if is_off_policy:
                policy_loss = policy_loss * self.config.off_policy_weight
                off_policy_rewards.append(reward)
            else:
                policy_loss = policy_loss * self.config.on_policy_weight
                on_policy_rewards.append(reward)
            
            total_loss += policy_loss
        
        # KL penalty
        if self.reference_model is not None:
            kl_div = self._compute_kl_divergence(batch)
            total_loss += self.config.kl_coef * kl_div
            metrics["kl_loss"] = kl_div.item() if torch.is_tensor(kl_div) else kl_div
        
        # Metriche
        if on_policy_rewards:
            metrics["on_policy_reward_mean"] = sum(on_policy_rewards) / len(on_policy_rewards)
        if off_policy_rewards:
            metrics["off_policy_reward_mean"] = sum(off_policy_rewards) / len(off_policy_rewards)
        
        avg_loss = total_loss / len(batch)
        metrics["policy_loss"] = avg_loss.item() if torch.is_tensor(avg_loss) else avg_loss
        
        return avg_loss, metrics
    
    def _compute_kl_divergence(self, batch: List[Dict[str, Any]]) -> torch.Tensor:
        """Calcola KL divergence tra policy corrente e reference."""
        kl_total = 0.0
        
        for item in batch:
            prompt = item["prompt"]
            response = item["response"]
            
            current_log_prob = self._compute_log_prob(prompt, response)
            
            # Reference log prob
            with torch.no_grad():
                ref_log_prob = self._compute_log_prob_with_model(
                    self.reference_model, prompt, response
                )
            
            kl = current_log_prob - ref_log_prob
            kl_total += kl
        
        return torch.tensor(kl_total / len(batch))
    
    def _compute_log_prob_with_model(
        self,
        model: PreTrainedModel,
        prompt: str,
        response: str
    ) -> float:
        """Calcola log prob con un modello specifico."""
        full_text = prompt + response
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        return token_log_probs.mean().item()
    
    def train_step(
        self,
        prompts: List[str],
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Esegue un training step.
        
        Args:
            prompts: Lista di prompt
            optimizer: Ottimizzatore
            
        Returns:
            Metriche del training step
        """
        self.model.train()
        
        # 1. Genera on-policy
        on_policy_data = self.generate_on_policy(prompts)
        
        # 2. Mix con off-policy
        batch = self.data_mixer.sample_mixed_batch(
            on_policy_data,
            batch_size=len(prompts) * self.config.num_generations
        )
        
        # 3. Calcola loss
        loss, metrics = self.compute_grpo_loss(batch)
        
        # 4. Backward
        optimizer.zero_grad()
        if torch.is_tensor(loss) and loss.requires_grad:
            loss.backward()
            optimizer.step()
        
        # 5. Aggiorna statistiche
        self.training_stats["total_steps"] += 1
        self.training_stats["policy_loss"].append(metrics["policy_loss"])
        
        return metrics
    
    def train(
        self,
        train_dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        output_dir: str = "./checkpoints",
    ) -> Dict[str, Any]:
        """
        Training loop principale.
        
        Args:
            train_dataset: Dataset di training
            num_epochs: Numero di epoche
            batch_size: Dimensione batch
            learning_rate: Learning rate
            output_dir: Directory per checkpoint
            
        Returns:
            Metriche finali
        """
        logger.info(f"Inizio training LUFFY - {num_epochs} epochs")
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # DataLoader
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        all_metrics = []
        
        for epoch in range(num_epochs):
            epoch_metrics = []
            
            for batch_idx, batch in enumerate(dataloader):
                prompts = batch["prompt"] if isinstance(batch, dict) else batch
                if isinstance(prompts, torch.Tensor):
                    prompts = prompts.tolist()
                
                metrics = self.train_step(prompts, optimizer)
                epoch_metrics.append(metrics)
                
                if batch_idx % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs} - "
                        f"Step {batch_idx} - "
                        f"Loss: {metrics['policy_loss']:.4f} - "
                        f"On-policy reward: {metrics['on_policy_reward_mean']:.4f}"
                    )
            
            # Statistiche epoca
            avg_loss = sum(m["policy_loss"] for m in epoch_metrics) / len(epoch_metrics)
            avg_on_policy = sum(m["on_policy_reward_mean"] for m in epoch_metrics) / len(epoch_metrics)
            
            logger.info(
                f"Epoch {epoch+1} completata - "
                f"Avg Loss: {avg_loss:.4f} - "
                f"Avg On-Policy Reward: {avg_on_policy:.4f}"
            )
            
            all_metrics.extend(epoch_metrics)
        
        # Salva modello
        self.save_model(output_dir)
        
        return {
            "total_steps": self.training_stats["total_steps"],
            "final_loss": all_metrics[-1]["policy_loss"] if all_metrics else 0,
            "metrics": all_metrics,
        }
    
    def save_model(self, output_dir: str) -> None:
        """Salva modello e configurazione."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Salva modello
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Salva config LUFFY
        config_dict = {
            "mode": self.config.mode.value,
            "off_policy_source": self.config.off_policy_source,
            "off_policy_weight": self.config.off_policy_weight,
            "on_policy_weight": self.config.on_policy_weight,
            "kl_coef": self.config.kl_coef,
            "clip_range": self.config.clip_range,
        }
        
        with open(os.path.join(output_dir, "luffy_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Modello salvato in: {output_dir}")


# =============================================================================
# Factory functions
# =============================================================================

def create_luffy_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: Dict[str, Any],
    reward_fn: Optional[Callable] = None,
) -> LuffyTrainer:
    """
    Factory per creare LuffyTrainer da config dict.
    
    Args:
        model: Modello
        tokenizer: Tokenizer
        config: Configurazione da YAML
        reward_fn: Funzione di reward opzionale
        
    Returns:
        LuffyTrainer configurato
    """
    luffy_config = config.get("luffy", {})
    
    mode_str = luffy_config.get("mode", "luffy")
    try:
        mode = OffPolicyMode(mode_str.lower())
    except ValueError:
        logger.warning(f"Mode '{mode_str}' non valido, uso 'luffy'")
        mode = OffPolicyMode.LUFFY
    
    trainer_config = LuffyConfig(
        mode=mode,
        off_policy_source=luffy_config.get("off_policy_source", "deepseek-r1"),
        off_policy_weight=luffy_config.get("off_policy_weight", 0.5),
        on_policy_weight=luffy_config.get("on_policy_weight", 0.5),
        temperature=luffy_config.get("temperature", 0.7),
        num_generations=luffy_config.get("num_generations", 4),
        max_new_tokens=luffy_config.get("max_new_tokens", 2048),
        kl_coef=luffy_config.get("kl_coef", 0.05),
        clip_range=luffy_config.get("clip_range", 0.2),
    )
    
    return LuffyTrainer(
        model=model,
        tokenizer=tokenizer,
        config=trainer_config,
        reward_fn=reward_fn,
    )

