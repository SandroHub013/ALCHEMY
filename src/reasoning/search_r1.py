"""
Search-R1: Reasoning with Search Integration.

Implementazione ispirata a Search-R1 e DeepSeek-R1 per ragionamento
con integrazione di ricerca web e knowledge base.

Search-R1 permette al modello di:
- Interagire con motori di ricerca durante il ragionamento
- Recuperare informazioni rilevanti in tempo reale
- Combinare reasoning e retrieval in modo fluido

Riferimenti:
- Search-R1: https://github.com/PeterGriffinJin/Search-R1
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- veRL: https://github.com/volcengine/verl
"""

from typing import Optional, Dict, Any, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import re
import asyncio
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)


# =============================================================================
# Search Engine Abstraction
# =============================================================================

class SearchEngineType(str, Enum):
    """Tipi di search engine supportati."""
    VECTOR = "vector"       # Vector similarity search (locale)
    BM25 = "bm25"          # BM25 ranking (locale)
    WEB = "web"            # Web search API
    HYBRID = "hybrid"      # Combinazione vector + BM25


class SearchEngine(ABC):
    """Interfaccia astratta per search engine."""
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Esegue una ricerca.
        
        Args:
            query: Query di ricerca
            top_k: Numero di risultati da ritornare
            
        Returns:
            Lista di risultati con 'content', 'score', 'metadata'
        """
        pass
    
    @abstractmethod
    async def search_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Versione asincrona della ricerca."""
        pass


class VectorSearchEngine(SearchEngine):
    """Search engine basato su vector similarity."""
    
    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        index: Optional[Any] = None,
        documents: Optional[List[str]] = None,
    ):
        """
        Inizializza vector search engine.
        
        Args:
            embedding_model: Modello per generare embeddings
            index: Indice FAISS o simile
            documents: Lista di documenti indicizzati
        """
        self.embedding_model = embedding_model
        self.index = index
        self.documents = documents or []
        
        # Lazy init del modello di embedding
        self._lazy_init = False
    
    def _ensure_init(self) -> None:
        """Inizializza componenti se necessario."""
        if self._lazy_init:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np
            
            if self.embedding_model is None:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            
            if self.index is None and self.documents:
                # Crea indice FAISS
                embeddings = self.embedding_model.encode(self.documents)
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)  # Inner product
                faiss.normalize_L2(embeddings)
                self.index.add(embeddings.astype(np.float32))
            
            self._lazy_init = True
        except ImportError as e:
            logger.warning(f"Dipendenze mancanti per VectorSearchEngine: {e}")
    
    def add_documents(self, documents: List[str]) -> None:
        """Aggiunge documenti all'indice."""
        self._ensure_init()
        
        import numpy as np
        
        self.documents.extend(documents)
        
        embeddings = self.embedding_model.encode(documents)
        import faiss
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Esegue ricerca vector similarity."""
        self._ensure_init()
        
        if self.index is None or not self.documents:
            return []
        
        import numpy as np
        import faiss
        
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(
            query_embedding.astype(np.float32), 
            min(top_k, len(self.documents))
        )
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(self.documents):
                results.append({
                    "content": self.documents[idx],
                    "score": float(score),
                    "rank": i + 1,
                    "metadata": {"index": int(idx)},
                })
        
        return results
    
    async def search_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Versione asincrona (wrapper)."""
        return self.search(query, top_k)


class BM25SearchEngine(SearchEngine):
    """Search engine basato su BM25."""
    
    def __init__(self, documents: Optional[List[str]] = None):
        """
        Inizializza BM25 search engine.
        
        Args:
            documents: Lista di documenti da indicizzare
        """
        self.documents = documents or []
        self.bm25 = None
        self.tokenized_corpus = None
        
        if documents:
            self._build_index(documents)
    
    def _build_index(self, documents: List[str]) -> None:
        """Costruisce l'indice BM25."""
        try:
            from rank_bm25 import BM25Okapi
            
            # Tokenizzazione semplice
            self.tokenized_corpus = [doc.lower().split() for doc in documents]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            self.documents = documents
        except ImportError:
            logger.warning("rank-bm25 non installato. Installa con: pip install rank-bm25")
    
    def add_documents(self, documents: List[str]) -> None:
        """Aggiunge documenti e ricostruisce l'indice."""
        all_docs = self.documents + documents
        self._build_index(all_docs)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Esegue ricerca BM25."""
        if self.bm25 is None:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "content": self.documents[idx],
                "score": float(scores[idx]),
                "rank": rank + 1,
                "metadata": {"index": idx},
            })
        
        return results
    
    async def search_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Versione asincrona (wrapper)."""
        return self.search(query, top_k)


class HybridSearchEngine(SearchEngine):
    """Search engine ibrido (vector + BM25)."""
    
    def __init__(
        self,
        documents: Optional[List[str]] = None,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ):
        """
        Inizializza hybrid search engine.
        
        Args:
            documents: Lista di documenti
            vector_weight: Peso per vector search
            bm25_weight: Peso per BM25
        """
        self.vector_engine = VectorSearchEngine(documents=documents)
        self.bm25_engine = BM25SearchEngine(documents=documents)
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
    
    def add_documents(self, documents: List[str]) -> None:
        """Aggiunge documenti a entrambi gli indici."""
        self.vector_engine.add_documents(documents)
        self.bm25_engine.add_documents(documents)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Esegue ricerca ibrida con fusione dei risultati."""
        vector_results = self.vector_engine.search(query, top_k * 2)
        bm25_results = self.bm25_engine.search(query, top_k * 2)
        
        # Reciprocal Rank Fusion
        doc_scores: Dict[str, float] = {}
        
        k = 60  # Costante RRF
        
        for rank, result in enumerate(vector_results):
            content = result["content"]
            doc_scores[content] = doc_scores.get(content, 0) + self.vector_weight / (k + rank + 1)
        
        for rank, result in enumerate(bm25_results):
            content = result["content"]
            doc_scores[content] = doc_scores.get(content, 0) + self.bm25_weight / (k + rank + 1)
        
        # Sort by fused score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (content, score) in enumerate(sorted_docs[:top_k]):
            results.append({
                "content": content,
                "score": score,
                "rank": rank + 1,
                "metadata": {"fusion_method": "rrf"},
            })
        
        return results
    
    async def search_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Versione asincrona."""
        return self.search(query, top_k)


# =============================================================================
# Search-R1 Configuration
# =============================================================================

@dataclass
class SearchR1Config:
    """
    Configurazione per Search-R1 trainer.
    
    Attributes:
        search_engine_type: Tipo di search engine
        max_search_calls: Massimo numero di ricerche per generazione
        search_token: Token speciale per attivare ricerca
        context_window: Numero di risultati da includere nel contesto
        temperature: Temperatura per generazione
        max_new_tokens: Massimo token generati
        use_cot: Usa Chain-of-Thought prompting
    """
    
    # Search configuration
    search_engine_type: SearchEngineType = SearchEngineType.HYBRID
    max_search_calls: int = 3
    search_token: str = "<search>"
    end_search_token: str = "</search>"
    context_token: str = "<context>"
    end_context_token: str = "</context>"
    context_window: int = 3
    
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 2048
    
    # Reasoning parameters
    use_cot: bool = True  # Chain-of-Thought
    use_reflection: bool = True  # Self-reflection dopo ricerca
    
    # RL parameters (per training)
    kl_coef: float = 0.05
    reward_search_bonus: float = 0.1  # Bonus per ricerca efficace
    reward_correctness_weight: float = 0.7
    reward_reasoning_weight: float = 0.3


# =============================================================================
# Reasoning with Search
# =============================================================================

class ReasoningWithSearch:
    """
    Classe per ragionamento con integrazione di ricerca.
    
    Permette al modello di:
    1. Generare pensieri iniziali
    2. Decidere quando cercare informazioni
    3. Incorporare risultati nel ragionamento
    4. Continuare il ragionamento con il nuovo contesto
    
    Esempio:
        ```python
        reasoner = ReasoningWithSearch(
            model=model,
            tokenizer=tokenizer,
            search_engine=HybridSearchEngine(documents=my_docs)
        )
        
        response = reasoner.reason("Qual è la capitale più popolosa d'Europa?")
        print(response["final_answer"])
        print(response["reasoning_trace"])
        ```
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        search_engine: SearchEngine,
        config: Optional[SearchR1Config] = None,
    ):
        """
        Inizializza reasoner con search.
        
        Args:
            model: Modello per generazione
            tokenizer: Tokenizer
            search_engine: Engine per ricerca
            config: Configurazione
        """
        self.model = model
        self.tokenizer = tokenizer
        self.search_engine = search_engine
        self.config = config or SearchR1Config()
        
        # Aggiungi token speciali se necessario
        self._ensure_special_tokens()
    
    def _ensure_special_tokens(self) -> None:
        """Assicura che i token speciali siano nel tokenizer."""
        special_tokens = [
            self.config.search_token,
            self.config.end_search_token,
            self.config.context_token,
            self.config.end_context_token,
        ]
        
        # Verifica quali token mancano
        new_tokens = []
        for token in special_tokens:
            if token not in self.tokenizer.get_vocab():
                new_tokens.append(token)
        
        if new_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
            # Resize model embeddings se necessario
            if hasattr(self.model, "resize_token_embeddings"):
                self.model.resize_token_embeddings(len(self.tokenizer))
    
    def _build_reasoning_prompt(self, question: str) -> str:
        """Costruisce il prompt per reasoning con search."""
        if self.config.use_cot:
            prompt = f"""You are a reasoning assistant with access to a search engine.

When you need to search for information, use the special tokens:
{self.config.search_token}your search query{self.config.end_search_token}

After searching, you will receive context in:
{self.config.context_token}search results{self.config.end_context_token}

Think step by step and search when needed.

Question: {question}

Let me think about this step by step."""
        else:
            prompt = f"Question: {question}\n\nAnswer:"
        
        return prompt
    
    def _extract_search_queries(self, text: str) -> List[str]:
        """Estrae query di ricerca dal testo generato."""
        pattern = f"{re.escape(self.config.search_token)}(.*?){re.escape(self.config.end_search_token)}"
        matches = re.findall(pattern, text, re.DOTALL)
        return [m.strip() for m in matches]
    
    def _inject_search_results(
        self,
        text: str,
        query: str,
        results: List[Dict[str, Any]],
    ) -> str:
        """Inietta risultati di ricerca nel testo."""
        # Formatta risultati
        formatted_results = []
        for i, result in enumerate(results[:self.config.context_window]):
            formatted_results.append(f"[{i+1}] {result['content'][:500]}")
        
        context = "\n".join(formatted_results)
        
        # Sostituisci search token con context
        search_pattern = f"{self.config.search_token}{re.escape(query)}{self.config.end_search_token}"
        replacement = (
            f"{self.config.search_token}{query}{self.config.end_search_token}\n"
            f"{self.config.context_token}\n{context}\n{self.config.end_context_token}"
        )
        
        return re.sub(search_pattern, replacement, text, count=1)
    
    def reason(
        self,
        question: str,
        max_iterations: int = 5,
    ) -> Dict[str, Any]:
        """
        Esegue ragionamento con ricerca iterativa.
        
        Args:
            question: Domanda da rispondere
            max_iterations: Massimo iterazioni di generate-search
            
        Returns:
            Dizionario con answer, reasoning_trace, search_queries
        """
        prompt = self._build_reasoning_prompt(question)
        
        reasoning_trace = []
        search_queries = []
        search_results_all = []
        
        current_text = prompt
        search_count = 0
        
        for iteration in range(max_iterations):
            # Genera
            inputs = self.tokenizer(
                current_text,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens // max_iterations,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=False
            )
            
            reasoning_trace.append({
                "iteration": iteration,
                "generated": generated,
            })
            
            # Controlla se c'è una search query
            queries = self._extract_search_queries(generated)
            
            if queries and search_count < self.config.max_search_calls:
                for query in queries:
                    # Esegui ricerca
                    results = self.search_engine.search(query, self.config.context_window)
                    
                    search_queries.append(query)
                    search_results_all.append(results)
                    search_count += 1
                    
                    # Inietta risultati
                    generated = self._inject_search_results(generated, query, results)
                
                # Continua generazione
                current_text = current_text + generated
            else:
                # Nessuna search o limite raggiunto, termina
                current_text = current_text + generated
                break
        
        # Estrai risposta finale
        final_answer = self._extract_final_answer(current_text)
        
        return {
            "question": question,
            "final_answer": final_answer,
            "reasoning_trace": reasoning_trace,
            "search_queries": search_queries,
            "search_results": search_results_all,
            "full_text": current_text,
        }
    
    def _extract_final_answer(self, text: str) -> str:
        """Estrae la risposta finale dal testo."""
        # Cerca pattern comuni per la risposta
        patterns = [
            r"(?:Therefore|Thus|So|Hence|In conclusion),?\s*(.+?)(?:\.|$)",
            r"(?:The answer is|Answer:)\s*(.+?)(?:\.|$)",
            r"\*\*Answer\*\*:?\s*(.+?)(?:\.|$)",
            r"\\boxed\{(.+?)\}",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Fallback: ultime righe
        lines = text.strip().split("\n")
        return lines[-1] if lines else text


# =============================================================================
# Search-R1 Trainer
# =============================================================================

class SearchR1Trainer:
    """
    Trainer per Search-R1 con reinforcement learning.
    
    Addestra il modello a:
    1. Decidere quando cercare
    2. Formulare query efficaci
    3. Utilizzare i risultati nel ragionamento
    
    Usa RL per ottimizzare la policy di search + reasoning.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        search_engine: SearchEngine,
        config: SearchR1Config,
        reward_fn: Optional[Callable] = None,
        reference_model: Optional[PreTrainedModel] = None,
    ):
        """
        Inizializza Search-R1 Trainer.
        
        Args:
            model: Modello da addestrare
            tokenizer: Tokenizer
            search_engine: Search engine
            config: Configurazione
            reward_fn: Funzione di reward custom
            reference_model: Modello di riferimento per KL
        """
        self.model = model
        self.tokenizer = tokenizer
        self.search_engine = search_engine
        self.config = config
        self.reward_fn = reward_fn or self._default_reward
        self.reference_model = reference_model
        
        # Reasoner
        self.reasoner = ReasoningWithSearch(
            model=model,
            tokenizer=tokenizer,
            search_engine=search_engine,
            config=config,
        )
        
        # Stats
        self.training_stats = {
            "total_steps": 0,
            "avg_search_calls": [],
            "avg_reward": [],
        }
        
        logger.info("SearchR1Trainer inizializzato")
    
    def _default_reward(
        self,
        question: str,
        answer: str,
        reference: Optional[str] = None,
        search_used: bool = False,
    ) -> float:
        """Reward di default per Search-R1."""
        reward = 0.0
        
        # Reward per risposta non vuota
        if len(answer.strip()) > 10:
            reward += 0.2
        
        # Bonus per uso efficace della ricerca
        if search_used:
            reward += self.config.reward_search_bonus
        
        # Reward per match con reference
        if reference:
            answer_lower = answer.lower()
            reference_lower = reference.lower()
            
            # Exact match
            if reference_lower in answer_lower:
                reward += 0.5
            else:
                # Partial match (parole chiave)
                ref_words = set(reference_lower.split())
                ans_words = set(answer_lower.split())
                overlap = len(ref_words & ans_words) / max(len(ref_words), 1)
                reward += 0.3 * overlap
        
        # Reward per ragionamento strutturato
        if "step" in answer.lower() or "therefore" in answer.lower():
            reward += 0.1
        
        return min(1.0, reward)
    
    def compute_reward(
        self,
        result: Dict[str, Any],
        reference: Optional[str] = None,
    ) -> float:
        """
        Calcola reward per un risultato di reasoning.
        
        Args:
            result: Output da ReasoningWithSearch.reason()
            reference: Risposta di riferimento
            
        Returns:
            Reward value
        """
        return self.reward_fn(
            question=result["question"],
            answer=result["final_answer"],
            reference=reference,
            search_used=len(result["search_queries"]) > 0,
        )
    
    def train_step(
        self,
        questions: List[str],
        references: Optional[List[str]],
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Esegue un training step.
        
        Args:
            questions: Lista di domande
            references: Risposte di riferimento (opzionale)
            optimizer: Ottimizzatore
            
        Returns:
            Metriche del training step
        """
        self.model.train()
        
        total_reward = 0.0
        total_search_calls = 0
        results_batch = []
        
        # Genera risposte con search
        for i, question in enumerate(questions):
            result = self.reasoner.reason(question)
            
            reference = references[i] if references else None
            reward = self.compute_reward(result, reference)
            
            total_reward += reward
            total_search_calls += len(result["search_queries"])
            
            results_batch.append({
                "result": result,
                "reward": reward,
            })
        
        # Calcola loss (semplificato - in produzione usare PPO/GRPO completo)
        loss = self._compute_policy_loss(results_batch)
        
        # Backward
        optimizer.zero_grad()
        if torch.is_tensor(loss) and loss.requires_grad:
            loss.backward()
            optimizer.step()
        
        # Metriche
        n = len(questions)
        metrics = {
            "avg_reward": total_reward / n,
            "avg_search_calls": total_search_calls / n,
            "loss": loss.item() if torch.is_tensor(loss) else loss,
        }
        
        self.training_stats["total_steps"] += 1
        self.training_stats["avg_reward"].append(metrics["avg_reward"])
        self.training_stats["avg_search_calls"].append(metrics["avg_search_calls"])
        
        return metrics
    
    def _compute_policy_loss(
        self,
        results_batch: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """
        Calcola policy loss per il batch.
        
        Args:
            results_batch: Lista di risultati con reward
            
        Returns:
            Loss tensor
        """
        # Normalizza rewards (relative)
        rewards = [r["reward"] for r in results_batch]
        mean_reward = sum(rewards) / len(rewards)
        std_reward = max(
            (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5,
            1e-8
        )
        
        advantages = [(r - mean_reward) / std_reward for r in rewards]
        
        # Loss come negative advantage (semplificato)
        # In produzione: usare log probs e importance sampling
        loss = -sum(advantages) / len(advantages)
        
        return torch.tensor(loss, requires_grad=True)
    
    def train(
        self,
        train_data: List[Dict[str, str]],
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        output_dir: str = "./checkpoints",
    ) -> Dict[str, Any]:
        """
        Training loop completo.
        
        Args:
            train_data: Lista di {"question": ..., "answer": ...}
            num_epochs: Numero di epoche
            batch_size: Batch size
            learning_rate: Learning rate
            output_dir: Directory output
            
        Returns:
            Metriche finali
        """
        logger.info(f"Inizio training Search-R1 - {num_epochs} epochs")
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        all_metrics = []
        
        for epoch in range(num_epochs):
            epoch_metrics = []
            
            # Batch training data
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                
                questions = [item["question"] for item in batch]
                references = [item.get("answer") for item in batch]
                
                metrics = self.train_step(questions, references, optimizer)
                epoch_metrics.append(metrics)
                
                if (i // batch_size) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs} - "
                        f"Step {i//batch_size} - "
                        f"Reward: {metrics['avg_reward']:.4f} - "
                        f"Searches: {metrics['avg_search_calls']:.2f}"
                    )
            
            # Epoch stats
            avg_reward = sum(m["avg_reward"] for m in epoch_metrics) / len(epoch_metrics)
            logger.info(f"Epoch {epoch+1} completata - Avg Reward: {avg_reward:.4f}")
            
            all_metrics.extend(epoch_metrics)
        
        # Salva
        self.save_model(output_dir)
        
        return {
            "total_steps": self.training_stats["total_steps"],
            "final_avg_reward": all_metrics[-1]["avg_reward"] if all_metrics else 0,
            "metrics": all_metrics,
        }
    
    def save_model(self, output_dir: str) -> None:
        """Salva modello e configurazione."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        config_dict = {
            "search_engine_type": self.config.search_engine_type.value,
            "max_search_calls": self.config.max_search_calls,
            "context_window": self.config.context_window,
            "use_cot": self.config.use_cot,
        }
        
        with open(os.path.join(output_dir, "search_r1_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Modello salvato in: {output_dir}")


# =============================================================================
# Factory functions
# =============================================================================

def create_search_engine(
    engine_type: str,
    documents: Optional[List[str]] = None,
    **kwargs
) -> SearchEngine:
    """
    Factory per creare search engine.
    
    Args:
        engine_type: Tipo di engine ("vector", "bm25", "hybrid")
        documents: Documenti da indicizzare
        **kwargs: Parametri addizionali
        
    Returns:
        SearchEngine configurato
    """
    engine_type = engine_type.lower()
    
    if engine_type == "vector":
        return VectorSearchEngine(documents=documents, **kwargs)
    elif engine_type == "bm25":
        return BM25SearchEngine(documents=documents)
    elif engine_type == "hybrid":
        return HybridSearchEngine(
            documents=documents,
            vector_weight=kwargs.get("vector_weight", 0.5),
            bm25_weight=kwargs.get("bm25_weight", 0.5),
        )
    else:
        raise ValueError(f"Engine type '{engine_type}' non supportato")


def create_search_r1_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: Dict[str, Any],
    documents: Optional[List[str]] = None,
) -> SearchR1Trainer:
    """
    Factory per creare SearchR1Trainer da config dict.
    
    Args:
        model: Modello
        tokenizer: Tokenizer
        config: Configurazione da YAML
        documents: Documenti per search engine
        
    Returns:
        SearchR1Trainer configurato
    """
    search_config = config.get("search_r1", {})
    
    engine_type = search_config.get("search_engine_type", "hybrid")
    try:
        engine_enum = SearchEngineType(engine_type.lower())
    except ValueError:
        engine_enum = SearchEngineType.HYBRID
    
    trainer_config = SearchR1Config(
        search_engine_type=engine_enum,
        max_search_calls=search_config.get("max_search_calls", 3),
        context_window=search_config.get("context_window", 3),
        temperature=search_config.get("temperature", 0.7),
        max_new_tokens=search_config.get("max_new_tokens", 2048),
        use_cot=search_config.get("use_cot", True),
    )
    
    # Crea search engine
    search_engine = create_search_engine(engine_type, documents)
    
    return SearchR1Trainer(
        model=model,
        tokenizer=tokenizer,
        search_engine=search_engine,
        config=trainer_config,
    )

