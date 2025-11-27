"""
Modulo per la gestione dei dataset con PyTorch Lightning DataModule.

Supporta:
- Single-source training (retrocompatibile)
- Multi-source training con Data Mixing per modelli generalisti
- Formattazione unificata ChatML per evitare Catastrophic Forgetting
"""

from typing import Optional, Dict, Any, List, Callable, Literal
from dataclasses import dataclass, field
import logging
import random
from pathlib import Path

from datasets import load_dataset, Dataset, DatasetDict, interleave_datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


# =============================================================================
# RAG / CONTEXTUAL TRAINING CONFIGURATION
# =============================================================================

# Percentuale di esempi con contesto RAG (RAFT strategy)
RAG_CONTEXT_PERCENTAGE = 0.20  # 20% degli esempi

# Template per contesto sintetico (usato quando non c'è contesto reale)
SYNTHETIC_CONTEXT_TEMPLATES = [
    "According to the documentation, {topic} is defined as follows: {snippet}",
    "From the knowledge base: {snippet}. This relates to {topic}.",
    "Reference material states: {snippet}",
    "The following information is relevant: {snippet}. Key concept: {topic}.",
    "Documentation excerpt: {snippet}",
]

def generate_synthetic_context(instruction: str, response: str) -> str:
    """
    Genera contesto sintetico per training RAG.
    
    Estrae frasi dalla risposta per simulare un contesto recuperato.
    Questo insegna al modello a cercare informazioni nel contesto.
    
    Args:
        instruction: La domanda
        response: La risposta (da cui estraiamo il contesto)
        
    Returns:
        Contesto sintetico
    """
    # Estrai parole chiave dall'istruzione
    words = instruction.split()
    topic = " ".join(words[:min(5, len(words))])
    
    # Prendi una parte della risposta come "snippet recuperato"
    response_sentences = response.replace("\n", " ").split(". ")
    
    if len(response_sentences) > 2:
        # Prendi le prime 2-3 frasi come contesto
        snippet = ". ".join(response_sentences[:min(3, len(response_sentences))])
    else:
        snippet = response[:min(500, len(response))]
    
    # Scegli un template casuale
    template = random.choice(SYNTHETIC_CONTEXT_TEMPLATES)
    
    try:
        context = template.format(topic=topic, snippet=snippet)
    except KeyError:
        context = f"Relevant information: {snippet}"
    
    return context


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

DatasetType = Literal["function_calling", "coding", "chat", "language", "instruction"]


@dataclass
class DatasetSourceConfig:
    """Configurazione per una singola fonte dati nel multi-source training."""
    
    name: str
    weight: float
    split: str = "train"
    type: DatasetType = "instruction"
    columns: Dict[str, str] = field(default_factory=dict)
    max_samples: Optional[int] = None
    config_name: Optional[str] = None  # Per dataset HuggingFace con subset


# =============================================================================
# FORMATTERS - Convertono diversi formati in ChatML unificato
# =============================================================================

class ChatMLFormatter:
    """
    Formattatore unificato per convertire qualsiasi formato in ChatML.
    
    ChatML Format:
    <|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant
    {assistant_message}<|im_end|>
    """
    
    CHATML_START = "<|im_start|>"
    CHATML_END = "<|im_end|>"
    
    @classmethod
    def format_message(cls, role: str, content: str) -> str:
        """Formatta un singolo messaggio in ChatML."""
        return f"{cls.CHATML_START}{role}\n{content}{cls.CHATML_END}"
    
    @classmethod
    def format_conversation(
        cls, 
        messages: List[Dict[str, str]], 
        system_message: Optional[str] = None
    ) -> str:
        """
        Formatta una conversazione completa in ChatML.
        
        Args:
            messages: Lista di dict con 'role' e 'content'
            system_message: Messaggio di sistema opzionale
            
        Returns:
            Stringa formattata in ChatML
        """
        parts = []
        
        if system_message:
            parts.append(cls.format_message("system", system_message))
        
        for msg in messages:
            role = msg.get("role", msg.get("from", "user"))
            content = msg.get("content", msg.get("value", ""))
            
            # Normalizza i ruoli
            if role in ("human", "user", "Human"):
                role = "user"
            elif role in ("gpt", "assistant", "Assistant", "bot"):
                role = "assistant"
            
            parts.append(cls.format_message(role, content))
        
        return "\n".join(parts)
    
    @classmethod
    def format_instruction_response(
        cls,
        instruction: str,
        response: str,
        system_message: Optional[str] = None,
        input_context: Optional[str] = None
    ) -> str:
        """
        Formatta una coppia instruction-response in ChatML.
        
        Args:
            instruction: L'istruzione/domanda
            response: La risposta
            system_message: Messaggio di sistema opzionale
            input_context: Contesto aggiuntivo (es. per dataset Alpaca)
        """
        # Combina instruction e input se presente
        user_content = instruction
        if input_context and input_context.strip():
            user_content = f"{instruction}\n\nContext:\n{input_context}"
        
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response}
        ]
        
        return cls.format_conversation(messages, system_message)
    
    @classmethod
    def format_with_rag_context(
        cls,
        instruction: str,
        response: str,
        retrieved_context: str,
    ) -> str:
        """
        Formatta con contesto RAG recuperato (RAFT Strategy).
        
        Questo formato insegna al modello a:
        1. Leggere il contesto fornito
        2. Usare le informazioni rilevanti
        3. Ammettere quando l'informazione non è nel contesto
        
        Args:
            instruction: La domanda dell'utente
            response: La risposta (che dovrebbe usare il contesto)
            retrieved_context: Il contesto recuperato dalla knowledge base
            
        Returns:
            Stringa formattata in ChatML con RAG context
        """
        system_message = (
            "You are an assistant. Use the following context to answer the question. "
            "If the answer is not in the context, say so.\n\n"
            f"Context:\n{retrieved_context}"
        )
        
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]
        
        return cls.format_conversation(messages, system_message)


def format_function_calling(example: Dict[str, Any], columns: Dict[str, str]) -> str:
    """
    Formatta un esempio di function calling in ChatML.
    
    I dataset di function calling tipicamente hanno:
    - Una conversazione con tool definitions
    - Chiamate a funzioni e risposte
    """
    # Gestisci diversi formati di function calling
    instruction_col = columns.get("instruction", "chat")
    response_col = columns.get("response", "answer")
    
    instruction = example.get(instruction_col, "")
    response = example.get(response_col, "")
    
    # Sistema message per function calling
    system_msg = (
        "You are a helpful assistant with access to functions. "
        "When a function is needed, call it using the proper format."
    )
    
    # Se l'istruzione contiene già una conversazione strutturata
    if isinstance(instruction, list):
        return ChatMLFormatter.format_conversation(instruction, system_msg)
    
    return ChatMLFormatter.format_instruction_response(
        instruction=str(instruction),
        response=str(response),
        system_message=system_msg
    )


def format_coding(
    example: Dict[str, Any], 
    columns: Dict[str, str],
    apply_rag: bool = False,
) -> str:
    """
    Formatta un esempio di coding in ChatML.
    
    Aggiunge system message specifico per coding.
    Se apply_rag=True, usa formato RAG con contesto sintetico.
    """
    instruction_col = columns.get("instruction", "instruction")
    response_col = columns.get("response", "output")
    
    instruction = example.get(instruction_col, "")
    response = example.get(response_col, "")
    
    instruction_str = str(instruction)
    response_str = str(response)
    
    # Applica RAG context se richiesto
    if apply_rag:
        context = generate_synthetic_context(instruction_str, response_str)
        return ChatMLFormatter.format_with_rag_context(
            instruction=instruction_str,
            response=response_str,
            retrieved_context=context,
        )
    
    system_msg = (
        "You are an expert programmer. Write clean, efficient, and well-documented code. "
        "Always explain your implementation choices when relevant."
    )
    
    return ChatMLFormatter.format_instruction_response(
        instruction=instruction_str,
        response=response_str,
        system_message=system_msg
    )


def format_chat(example: Dict[str, Any], columns: Dict[str, str]) -> str:
    """
    Formatta un esempio di chat/conversazione in ChatML.
    
    Supporta formato OpenHermes/ShareGPT con lista di messaggi.
    """
    # OpenHermes usa "conversations" come lista di messaggi
    conversations_col = columns.get("conversations", "conversations")
    
    if conversations_col in example:
        conversations = example[conversations_col]
        if isinstance(conversations, list):
            return ChatMLFormatter.format_conversation(conversations)
    
    # Fallback a formato instruction-response
    instruction_col = columns.get("instruction", "instruction")
    response_col = columns.get("response", "response")
    
    return ChatMLFormatter.format_instruction_response(
        instruction=str(example.get(instruction_col, "")),
        response=str(example.get(response_col, ""))
    )


def format_language(example: Dict[str, Any], columns: Dict[str, str]) -> str:
    """
    Formatta testo monolingue in formato ChatML pseudo-conversazionale.
    
    Usato per dataset di lingua (es. italiano) per mantenere competenza linguistica.
    """
    text_col = columns.get("text", "text")
    text = example.get(text_col, "")
    
    # Crea una pseudo-conversazione per continuare la generazione
    # Questo aiuta il modello a mantenere fluenza nella lingua
    if len(str(text)) > 100:
        # Splitta il testo e crea una domanda-risposta
        text_str = str(text)
        split_point = len(text_str) // 3  # Primo terzo come "prompt"
        
        # Trova un punto naturale di split (fine frase)
        for i in range(split_point, min(split_point + 200, len(text_str))):
            if text_str[i] in ".!?":
                split_point = i + 1
                break
        
        prompt = text_str[:split_point].strip()
        continuation = text_str[split_point:].strip()
        
        return ChatMLFormatter.format_instruction_response(
            instruction=f"Continua questo testo:\n\n{prompt}",
            response=continuation,
            system_message="Sei un assistente che scrive in italiano fluente e naturale."
        )
    
    # Testo troppo corto, usa come risposta semplice
    return ChatMLFormatter.format_instruction_response(
        instruction="Scrivi un breve testo informativo.",
        response=str(text),
        system_message="Sei un assistente che scrive in italiano fluente e naturale."
    )


def format_instruction(
    example: Dict[str, Any], 
    columns: Dict[str, str],
    apply_rag: bool = False,
) -> str:
    """
    Formatta un esempio instruction-response generico in ChatML.
    Se apply_rag=True, usa formato RAG con contesto.
    """
    instruction_col = columns.get("instruction", "instruction")
    response_col = columns.get("response", "response")
    input_col = columns.get("input", "input")
    context_col = columns.get("context", "context")
    
    instruction_str = str(example.get(instruction_col, ""))
    response_str = str(example.get(response_col, ""))
    
    # Applica RAG context se richiesto
    if apply_rag:
        # Prima controlla se c'è un campo context nel dataset
        if context_col in example and example.get(context_col):
            context = str(example.get(context_col))
        else:
            # Genera contesto sintetico
            context = generate_synthetic_context(instruction_str, response_str)
        
        return ChatMLFormatter.format_with_rag_context(
            instruction=instruction_str,
            response=response_str,
            retrieved_context=context,
        )
    
    return ChatMLFormatter.format_instruction_response(
        instruction=instruction_str,
        response=response_str,
        input_context=str(example.get(input_col, "")) if input_col in example else None
    )


# Registry dei formatters per tipo di dataset
FORMATTERS: Dict[DatasetType, Callable[[Dict[str, Any], Dict[str, str]], str]] = {
    "function_calling": format_function_calling,
    "coding": format_coding,
    "chat": format_chat,
    "language": format_language,
    "instruction": format_instruction,
}


# =============================================================================
# MULTI-SOURCE DATA MODULE
# =============================================================================

class MultiSourceDataModule(pl.LightningDataModule):
    """
    LightningDataModule per Multi-Source Training (Data Mixing).
    
    Carica e mescola più dataset con pesi configurabili per creare
    un modello generalista che sa fare coding, function calling e chat
    senza soffrire di Catastrophic Forgetting.
    
    Usa `datasets.interleave_datasets` per campionare proporzionalmente
    dai diversi dataset durante il training.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        sources: List[DatasetSourceConfig],
        output_format: str = "chatml",
        max_seq_length: int = 2048,
        val_split_percentage: float = 0.1,
        per_device_train_batch_size: int = 2,
        per_device_eval_batch_size: int = 2,
        num_workers: int = 4,
        seed: int = 42,
    ):
        """
        Inizializza il MultiSourceDataModule.
        
        Args:
            tokenizer: Tokenizer per la tokenizzazione
            sources: Lista di DatasetSourceConfig con le fonti dati
            output_format: Formato di output ("chatml", "alpaca", etc.)
            max_seq_length: Lunghezza massima sequenze
            val_split_percentage: Percentuale per validation set
            per_device_train_batch_size: Batch size per GPU (training)
            per_device_eval_batch_size: Batch size per GPU (eval)
            num_workers: Worker per DataLoader
            seed: Seed per riproducibilità
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.sources = sources
        self.output_format = output_format
        self.max_seq_length = max_seq_length
        self.val_split_percentage = val_split_percentage
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.num_workers = num_workers
        self.seed = seed
        
        self.train_dataset = None
        self.val_dataset = None
        
        # Valida i pesi
        total_weight = sum(s.weight for s in sources)
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"I pesi dei dataset sommano a {total_weight:.2f}, non 1.0. "
                "Verranno normalizzati automaticamente."
            )
    
    def _load_single_source(self, source: DatasetSourceConfig) -> Dataset:
        """
        Carica un singolo dataset e applica il formatter appropriato.
        
        Args:
            source: Configurazione della fonte dati
            
        Returns:
            Dataset formattato
        """
        logger.info(f"Caricamento dataset: {source.name} (weight: {source.weight:.2f})")
        
        # Carica il dataset
        try:
            if source.config_name:
                dataset = load_dataset(
                    source.name,
                    source.config_name,
                    split=source.split
                )
            else:
                dataset = load_dataset(source.name, split=source.split)
        except Exception as e:
            logger.error(f"Errore caricamento {source.name}: {e}")
            raise
        
        # Limita il numero di esempi se specificato
        if source.max_samples and len(dataset) > source.max_samples:
            dataset = dataset.shuffle(seed=self.seed).select(range(source.max_samples))
            logger.info(f"  Limitato a {source.max_samples} esempi")
        
        # Ottieni il formatter appropriato
        formatter = FORMATTERS.get(source.type, format_instruction)
        
        # Conta quanti esempi avranno RAG context
        rag_count = 0
        total_count = 0
        
        # Applica la formattazione con RAFT strategy (20% RAG context)
        def format_example(example: Dict[str, Any]) -> Dict[str, str]:
            nonlocal rag_count, total_count
            total_count += 1
            
            try:
                # Decidi se applicare RAG context (20% probabilità)
                apply_rag = random.random() < RAG_CONTEXT_PERCENTAGE
                
                # Solo alcuni formatter supportano RAG
                if apply_rag and source.type in ("coding", "instruction"):
                    rag_count += 1
                    # Usa il formatter con RAG
                    if source.type == "coding":
                        formatted_text = format_coding(example, source.columns, apply_rag=True)
                    else:
                        formatted_text = format_instruction(example, source.columns, apply_rag=True)
                else:
                    # Formatter normale
                    formatted_text = formatter(example, source.columns)
                
                return {"text": formatted_text, "_source": source.name}
            except Exception as e:
                logger.warning(f"Errore formattazione esempio: {e}")
                return {"text": "", "_source": source.name}
        
        formatted_dataset = dataset.map(
            format_example,
            remove_columns=dataset.column_names,
            desc=f"Formattazione {source.name}",
        )
        
        # Log RAG context stats
        if rag_count > 0:
            logger.info(f"  RAG Context applicato a {rag_count}/{total_count} esempi ({rag_count/total_count*100:.1f}%)")
        
        # Rimuovi esempi vuoti
        formatted_dataset = formatted_dataset.filter(
            lambda x: len(x["text"]) > 10,
            desc=f"Filtraggio {source.name}"
        )
        
        logger.info(f"  Caricati {len(formatted_dataset)} esempi da {source.name}")
        return formatted_dataset
    
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Carica tutti i dataset, li mescola e prepara per il training.
        
        Usa `interleave_datasets` per campionare proporzionalmente
        dai diversi dataset in base ai pesi configurati.
        """
        if stage == "fit" or stage is None:
            logger.info("=" * 60)
            logger.info("SETUP MULTI-SOURCE TRAINING")
            logger.info("=" * 60)
            
            # Carica tutti i dataset
            datasets_list = []
            for source in self.sources:
                ds = self._load_single_source(source)
                datasets_list.append(ds)
            
            # Calcola le probabilità normalizzate
            weights = [s.weight for s in self.sources]
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]
            
            logger.info(f"Probabilità di campionamento: {probabilities}")
            
            # Mescola i dataset con interleave_datasets
            # stopping_strategy="all_exhausted" assicura che tutti i dataset
            # vengano usati completamente
            interleaved_dataset = interleave_datasets(
                datasets_list,
                probabilities=probabilities,
                seed=self.seed,
                stopping_strategy="first_exhausted"  # Si ferma quando finisce il più piccolo
            )
            
            logger.info(f"Dataset mescolato: {len(interleaved_dataset)} esempi totali")
            
            # Split train/val
            split_dataset = interleaved_dataset.train_test_split(
                test_size=self.val_split_percentage,
                seed=self.seed,
            )
            
            train_data = split_dataset["train"]
            val_data = split_dataset["test"]
            
            # Tokenizza
            self.train_dataset = train_data.map(
                self._tokenize_function,
                batched=True,
                remove_columns=train_data.column_names,
                desc="Tokenizzazione training",
            )
            
            self.val_dataset = val_data.map(
                self._tokenize_function,
                batched=True,
                remove_columns=val_data.column_names,
                desc="Tokenizzazione validazione",
            )
            
            # Imposta formato PyTorch per tensori
            self.train_dataset.set_format("torch")
            self.val_dataset.set_format("torch")
            
            logger.info("=" * 60)
            logger.info(
                f"Dataset finale - Training: {len(self.train_dataset)} esempi, "
                f"Validazione: {len(self.val_dataset)} esempi"
            )
            logger.info("=" * 60)
    
    def _tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenizza i testi già formattati in ChatML.
        """
        texts = examples["text"]
        
        # Tokenizza
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        
        # Crea labels per causal LM
        labels = []
        for input_ids in tokenized["input_ids"]:
            label = [
                -100 if token_id == self.tokenizer.pad_token_id else token_id
                for token_id in input_ids
            ]
            labels.append(label)
        
        tokenized["labels"] = labels
        return tokenized
    
    def train_dataloader(self) -> DataLoader:
        """Crea il DataLoader per il training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Crea il DataLoader per la validazione."""
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.per_device_eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# =============================================================================
# LEGACY SINGLE-SOURCE DATA MODULE (Retrocompatibile)
# =============================================================================

class InstructionDataModule(pl.LightningDataModule):
    """
    LightningDataModule per dataset di instruction tuning (single-source).
    
    Supporta dataset da HuggingFace con formato instruction-response
    (es. databricks/databricks-dolly-15k, alpaca, ecc.).
    
    NOTE: Per multi-source training, usa MultiSourceDataModule.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        text_column: str = "instruction",
        response_column: str = "response",
        max_seq_length: int = 2048,
        train_split: str = "train",
        val_split: Optional[str] = None,
        val_split_percentage: float = 0.1,
        per_device_train_batch_size: int = 2,
        per_device_eval_batch_size: int = 2,
        gradient_accumulation_steps: int = 1,
        num_workers: int = 4,
        use_chatml: bool = True,  # Nuovo parametro per formato ChatML
    ):
        """
        Inizializza il DataModule.
        
        Args:
            tokenizer: Tokenizer per la tokenizzazione dei testi
            dataset_name: Nome del dataset su HuggingFace o path locale
            dataset_config: Configurazione specifica del dataset (opzionale)
            text_column: Nome della colonna con le istruzioni/prompt
            response_column: Nome della colonna con le risposte
            max_seq_length: Lunghezza massima delle sequenze tokenizzate
            train_split: Nome dello split di training
            val_split: Nome dello split di validazione (None = crea da train)
            val_split_percentage: Percentuale di train da usare come val (se val_split=None)
            per_device_train_batch_size: Batch size per GPU per training
            per_device_eval_batch_size: Batch size per GPU per validazione
            gradient_accumulation_steps: Passi di accumulazione gradienti
            num_workers: Numero di worker per DataLoader
            use_chatml: Se True, usa formato ChatML invece di Alpaca
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.text_column = text_column
        self.response_column = response_column
        self.max_seq_length = max_seq_length
        self.train_split = train_split
        self.val_split = val_split
        self.val_split_percentage = val_split_percentage
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_workers = num_workers
        self.use_chatml = use_chatml
        
        self.train_dataset = None
        self.val_dataset = None
    
    def prepare_data(self) -> None:
        """
        Scarica e prepara il dataset (chiamato solo su rank 0 in DDP).
        Non dovrebbe modificare lo stato (self).
        """
        logger.info(f"Preparazione dataset: {self.dataset_name}")
        
        # Carica il dataset (solo download, non tokenizzazione)
        if not self._is_local_path(self.dataset_name):
            load_dataset(
                self.dataset_name,
                name=self.dataset_config,
                split=None,
            )
        logger.info("Dataset scaricato e pronto")
    
    def _is_local_path(self, path: str) -> bool:
        """Verifica se il path è un file locale."""
        return Path(path).exists() or path.endswith(('.json', '.jsonl', '.csv', '.parquet'))
    
    def _load_local_dataset(self, file_path: str) -> DatasetDict:
        """Carica un dataset locale da file JSON, JSONL, CSV o Parquet."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File dataset non trovato: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.jsonl':
            dataset = load_dataset('json', data_files=str(file_path))
        elif suffix == '.json':
            dataset = load_dataset('json', data_files=str(file_path))
        elif suffix == '.csv':
            dataset = load_dataset('csv', data_files=str(file_path))
        elif suffix == '.parquet':
            dataset = load_dataset('parquet', data_files=str(file_path))
        else:
            raise ValueError(
                f"Formato file non supportato: {suffix}. "
                "Formati supportati: .json, .jsonl, .csv, .parquet"
            )
        
        if isinstance(dataset, Dataset):
            dataset = DatasetDict({"train": dataset})
        elif "train" not in dataset:
            first_key = list(dataset.keys())[0]
            dataset = DatasetDict({"train": dataset[first_key]})
        
        return dataset
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Carica e tokenizza il dataset per training/validazione."""
        if stage == "fit" or stage is None:
            logger.info("Setup dataset per training")
            
            if self._is_local_path(self.dataset_name):
                logger.info(f"Caricamento dataset locale: {self.dataset_name}")
                dataset_dict = self._load_local_dataset(self.dataset_name)
            else:
                logger.info(f"Caricamento dataset da HuggingFace: {self.dataset_name}")
                if self.dataset_config:
                    dataset_dict = load_dataset(
                        self.dataset_name,
                        name=self.dataset_config,
                    )
                else:
                    dataset_dict = load_dataset(self.dataset_name)
            
            if self.train_split in dataset_dict:
                train_data = dataset_dict[self.train_split]
            else:
                raise ValueError(
                    f"Split '{self.train_split}' non trovato nel dataset. "
                    f"Split disponibili: {list(dataset_dict.keys())}"
                )
            
            if self.val_split and self.val_split in dataset_dict:
                val_data = dataset_dict[self.val_split]
            elif self.val_split_percentage > 0:
                split_dict = train_data.train_test_split(
                    test_size=self.val_split_percentage,
                    seed=42,
                )
                train_data = split_dict["train"]
                val_data = split_dict["test"]
                logger.info(
                    f"Creato split validazione: {len(val_data)} esempi "
                    f"({self.val_split_percentage*100:.1f}% del training)"
                )
            else:
                val_data = None
            
            self.train_dataset = train_data.map(
                self._tokenize_function,
                batched=True,
                remove_columns=train_data.column_names,
                desc="Tokenizzazione training",
            )
            
            if val_data is not None:
                self.val_dataset = val_data.map(
                    self._tokenize_function,
                    batched=True,
                    remove_columns=val_data.column_names,
                    desc="Tokenizzazione validazione",
                )
            else:
                self.val_dataset = None
            
            logger.info(
                f"Dataset preparato - Training: {len(self.train_dataset)} esempi, "
                f"Validazione: {len(self.val_dataset) if self.val_dataset else 0} esempi"
            )
    
    def _tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenizza gli esempi del dataset."""
        instructions = examples[self.text_column]
        responses = examples[self.response_column]
        
        texts = []
        for instruction, response in zip(instructions, responses):
            if self.use_chatml:
                # Usa formato ChatML
                text = ChatMLFormatter.format_instruction_response(
                    instruction=str(instruction),
                    response=str(response)
                )
            else:
                # Formato legacy Alpaca
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            texts.append(text)
        
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        
        labels = []
        for i, input_ids in enumerate(tokenized["input_ids"]):
            label = [
                -100 if token_id == self.tokenizer.pad_token_id else token_id
                for token_id in input_ids
            ]
            labels.append(label)
        
        tokenized["labels"] = labels
        return tokenized
    
    def train_dataloader(self) -> DataLoader:
        """Crea il DataLoader per il training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Crea il DataLoader per la validazione."""
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.per_device_eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_data_module(
    tokenizer: PreTrainedTokenizer,
    config: Dict[str, Any],
) -> pl.LightningDataModule:
    """
    Factory function per creare il DataModule appropriato basandosi sulla config.
    
    Args:
        tokenizer: Tokenizer per la tokenizzazione
        config: Configurazione completa (da config.yaml)
        
    Returns:
        InstructionDataModule o MultiSourceDataModule in base alla config
    """
    datasets_config = config.get("datasets", {})
    multi_source_enabled = datasets_config.get("multi_source_enabled", False)
    
    if multi_source_enabled:
        # Multi-source training
        sources = []
        for source_cfg in datasets_config.get("sources", []):
            sources.append(DatasetSourceConfig(
                name=source_cfg["name"],
                weight=source_cfg.get("weight", 1.0),
                split=source_cfg.get("split", "train"),
                type=source_cfg.get("type", "instruction"),
                columns=source_cfg.get("columns", {}),
                max_samples=source_cfg.get("max_samples"),
                config_name=source_cfg.get("config_name"),
            ))
        
        if not sources:
            raise ValueError("Multi-source abilitato ma nessun dataset specificato in 'sources'")
        
        training_config = config.get("training", {})
        data_config = config.get("data", {})
        
        logger.info(f"Creazione MultiSourceDataModule con {len(sources)} dataset")
        
        return MultiSourceDataModule(
            tokenizer=tokenizer,
            sources=sources,
            output_format=datasets_config.get("output_format", "chatml"),
            max_seq_length=data_config.get("max_seq_length", 2048),
            val_split_percentage=data_config.get("val_split_percentage", 0.1),
            per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
            per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 2),
            num_workers=training_config.get("dataloader_num_workers", 4),
            seed=training_config.get("seed", 42),
        )
    else:
        # Single-source training (legacy)
        data_config = config.get("data", {})
        training_config = config.get("training", {})
        
        logger.info(f"Creazione InstructionDataModule (single-source)")
        
        return InstructionDataModule(
            tokenizer=tokenizer,
            dataset_name=data_config.get("dataset_name", "databricks/databricks-dolly-15k"),
            dataset_config=data_config.get("dataset_config"),
            text_column=data_config.get("text_column", "instruction"),
            response_column=data_config.get("response_column", "response"),
            max_seq_length=data_config.get("max_seq_length", 2048),
            train_split=data_config.get("train_split", "train"),
            val_split=data_config.get("val_split"),
            val_split_percentage=data_config.get("val_split_percentage", 0.1),
            per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
            per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 2),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
            num_workers=training_config.get("dataloader_num_workers", 4),
            use_chatml=True,  # Usa ChatML di default per consistenza
        )
