"""
Module for dataset management with PyTorch Lightning DataModule.

Supports:
- Single-source training (backward compatible)
- Multi-source training with Data Mixing for generalist models
- Unified ChatML formatting to avoid Catastrophic Forgetting
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

# Percentage of examples with RAG context (RAFT strategy)
RAG_CONTEXT_PERCENTAGE = 0.20  # 20% of examples

# Templates for synthetic context (used when there's no real context)
SYNTHETIC_CONTEXT_TEMPLATES = [
    "According to the documentation, {topic} is defined as follows: {snippet}",
    "From the knowledge base: {snippet}. This relates to {topic}.",
    "Reference material states: {snippet}",
    "The following information is relevant: {snippet}. Key concept: {topic}.",
    "Documentation excerpt: {snippet}",
]

def generate_synthetic_context(instruction: str, response: str) -> str:
    """
    Generate synthetic context for RAG training.
    
    Extracts sentences from the response to simulate retrieved context.
    This teaches the model to search for information in the context.
    
    Args:
        instruction: The question
        response: The answer (from which we extract the context)
        
    Returns:
        Synthetic context
    """
    # Extract keywords from instruction
    words = instruction.split()
    topic = " ".join(words[:min(5, len(words))])
    
    # Take a part of the response as "retrieved snippet"
    response_sentences = response.replace("\n", " ").split(". ")
    
    if len(response_sentences) > 2:
        # Take the first 2-3 sentences as context
        snippet = ". ".join(response_sentences[:min(3, len(response_sentences))])
    else:
        snippet = response[:min(500, len(response))]
    
    # Choose a random template
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
    """Configuration for a single data source in multi-source training."""
    
    name: str
    weight: float
    split: str = "train"
    type: DatasetType = "instruction"
    columns: Dict[str, str] = field(default_factory=dict)
    max_samples: Optional[int] = None
    config_name: Optional[str] = None  # For HuggingFace datasets with subsets


# =============================================================================
# FORMATTERS - Convert different formats to unified ChatML
# =============================================================================

class ChatMLFormatter:
    """
    Unified formatter to convert any format to ChatML.
    
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
        """Format a single message in ChatML."""
        return f"{cls.CHATML_START}{role}\n{content}{cls.CHATML_END}"
    
    @classmethod
    def format_conversation(
        cls, 
        messages: List[Dict[str, str]], 
        system_message: Optional[str] = None
    ) -> str:
        """
        Format a complete conversation in ChatML.
        
        Args:
            messages: List of dicts with 'role' and 'content'
            system_message: Optional system message
            
        Returns:
            String formatted in ChatML
        """
        parts = []
        
        if system_message:
            parts.append(cls.format_message("system", system_message))
        
        for msg in messages:
            role = msg.get("role", msg.get("from", "user"))
            content = msg.get("content", msg.get("value", ""))
            
            # Normalize roles
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
        Format an instruction-response pair in ChatML.
        
        Args:
            instruction: The instruction/question
            response: The response
            system_message: Optional system message
            input_context: Additional context (e.g., for Alpaca datasets)
        """
        # Combine instruction and input if present
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
        Format with retrieved RAG context (RAFT Strategy).
        
        This format teaches the model to:
        1. Read the provided context
        2. Use relevant information
        3. Admit when information is not in the context
        
        Args:
            instruction: The user's question
            response: The answer (which should use the context)
            retrieved_context: The context retrieved from knowledge base
            
        Returns:
            String formatted in ChatML with RAG context
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
    Format a function calling example in ChatML.
    
    Function calling datasets typically have:
    - A conversation with tool definitions
    - Function calls and responses
    """
    # Handle different function calling formats
    instruction_col = columns.get("instruction", "chat")
    response_col = columns.get("response", "answer")
    
    instruction = example.get(instruction_col, "")
    response = example.get(response_col, "")
    
    # System message for function calling
    system_msg = (
        "You are a helpful assistant with access to functions. "
        "When a function is needed, call it using the proper format."
    )
    
    # If instruction already contains a structured conversation
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
    Format a coding example in ChatML.
    
    Adds specific system message for coding.
    If apply_rag=True, uses RAG format with synthetic context.
    """
    instruction_col = columns.get("instruction", "instruction")
    response_col = columns.get("response", "output")
    
    instruction = example.get(instruction_col, "")
    response = example.get(response_col, "")
    
    instruction_str = str(instruction)
    response_str = str(response)
    
    # Apply RAG context if requested
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
    Format a chat/conversation example in ChatML.
    
    Supports OpenHermes/ShareGPT format with message lists.
    """
    # OpenHermes uses "conversations" as a list of messages
    conversations_col = columns.get("conversations", "conversations")
    
    if conversations_col in example:
        conversations = example[conversations_col]
        if isinstance(conversations, list):
            return ChatMLFormatter.format_conversation(conversations)
    
    # Fallback to instruction-response format
    instruction_col = columns.get("instruction", "instruction")
    response_col = columns.get("response", "response")
    
    return ChatMLFormatter.format_instruction_response(
        instruction=str(example.get(instruction_col, "")),
        response=str(example.get(response_col, ""))
    )


def format_language(example: Dict[str, Any], columns: Dict[str, str]) -> str:
    """
    Format monolingual text in pseudo-conversational ChatML format.
    
    Used for language datasets to maintain linguistic competence.
    """
    text_col = columns.get("text", "text")
    text = example.get(text_col, "")
    
    # Create a pseudo-conversation for continuation generation
    # This helps the model maintain fluency in the language
    if len(str(text)) > 100:
        # Split the text and create a question-answer
        text_str = str(text)
        split_point = len(text_str) // 3  # First third as "prompt"
        
        # Find a natural split point (end of sentence)
        for i in range(split_point, min(split_point + 200, len(text_str))):
            if text_str[i] in ".!?":
                split_point = i + 1
                break
        
        prompt = text_str[:split_point].strip()
        continuation = text_str[split_point:].strip()
        
        return ChatMLFormatter.format_instruction_response(
            instruction=f"Continue this text:\n\n{prompt}",
            response=continuation,
            system_message="You are an assistant that writes fluently and naturally."
        )
    
    # Text too short, use as simple response
    return ChatMLFormatter.format_instruction_response(
        instruction="Write a brief informative text.",
        response=str(text),
        system_message="You are an assistant that writes fluently and naturally."
    )


def format_instruction(
    example: Dict[str, Any], 
    columns: Dict[str, str],
    apply_rag: bool = False,
) -> str:
    """
    Format a generic instruction-response example in ChatML.
    If apply_rag=True, uses RAG format with context.
    """
    instruction_col = columns.get("instruction", "instruction")
    response_col = columns.get("response", "response")
    input_col = columns.get("input", "input")
    context_col = columns.get("context", "context")
    
    instruction_str = str(example.get(instruction_col, ""))
    response_str = str(example.get(response_col, ""))
    
    # Apply RAG context if requested
    if apply_rag:
        # First check if there's a context field in the dataset
        if context_col in example and example.get(context_col):
            context = str(example.get(context_col))
        else:
            # Generate synthetic context
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


# Registry of formatters by dataset type
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
    LightningDataModule for Multi-Source Training (Data Mixing).
    
    Loads and mixes multiple datasets with configurable weights to create
    a generalist model that can do coding, function calling, and chat
    without suffering from Catastrophic Forgetting.
    
    Uses `datasets.interleave_datasets` to sample proportionally
    from different datasets during training.
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
        Initialize the MultiSourceDataModule.
        
        Args:
            tokenizer: Tokenizer for tokenization
            sources: List of DatasetSourceConfig with data sources
            output_format: Output format ("chatml", "alpaca", etc.)
            max_seq_length: Maximum sequence length
            val_split_percentage: Percentage for validation set
            per_device_train_batch_size: Batch size per GPU (training)
            per_device_eval_batch_size: Batch size per GPU (eval)
            num_workers: Workers for DataLoader
            seed: Seed for reproducibility
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
        
        # Validate weights
        total_weight = sum(s.weight for s in sources)
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Dataset weights sum to {total_weight:.2f}, not 1.0. "
                "They will be automatically normalized."
            )
    
    def _load_single_source(self, source: DatasetSourceConfig) -> Dataset:
        """
        Load a single dataset and apply the appropriate formatter.
        
        Args:
            source: Data source configuration
            
        Returns:
            Formatted dataset
        """
        logger.info(f"Loading dataset: {source.name} (weight: {source.weight:.2f})")
        
        # Load the dataset
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
            logger.error(f"Error loading {source.name}: {e}")
            raise
        
        # Limit number of examples if specified
        if source.max_samples and len(dataset) > source.max_samples:
            dataset = dataset.shuffle(seed=self.seed).select(range(source.max_samples))
            logger.info(f"  Limited to {source.max_samples} examples")
        
        # Get the appropriate formatter
        formatter = FORMATTERS.get(source.type, format_instruction)
        
        # Count how many examples will have RAG context
        rag_count = 0
        total_count = 0
        
        # Apply formatting with RAFT strategy (20% RAG context)
        def format_example(example: Dict[str, Any]) -> Dict[str, str]:
            nonlocal rag_count, total_count
            total_count += 1
            
            try:
                # Decide whether to apply RAG context (20% probability)
                apply_rag = random.random() < RAG_CONTEXT_PERCENTAGE
                
                # Only some formatters support RAG
                if apply_rag and source.type in ("coding", "instruction"):
                    rag_count += 1
                    # Use formatter with RAG
                    if source.type == "coding":
                        formatted_text = format_coding(example, source.columns, apply_rag=True)
                    else:
                        formatted_text = format_instruction(example, source.columns, apply_rag=True)
                else:
                    # Normal formatter
                    formatted_text = formatter(example, source.columns)
                
                return {"text": formatted_text, "_source": source.name}
            except Exception as e:
                logger.warning(f"Error formatting example: {e}")
                return {"text": "", "_source": source.name}
        
        formatted_dataset = dataset.map(
            format_example,
            remove_columns=dataset.column_names,
            desc=f"Formatting {source.name}",
        )
        
        # Log RAG context stats
        if rag_count > 0:
            logger.info(f"  RAG Context applied to {rag_count}/{total_count} examples ({rag_count/total_count*100:.1f}%)")
        
        # Remove empty examples
        formatted_dataset = formatted_dataset.filter(
            lambda x: len(x["text"]) > 10,
            desc=f"Filtering {source.name}"
        )
        
        logger.info(f"  Loaded {len(formatted_dataset)} examples from {source.name}")
        return formatted_dataset
    
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load all datasets, mix them, and prepare for training.
        
        Uses `interleave_datasets` to sample proportionally
        from different datasets based on configured weights.
        """
        if stage == "fit" or stage is None:
            logger.info("=" * 60)
            logger.info("SETUP MULTI-SOURCE TRAINING")
            logger.info("=" * 60)
            
            # Load all datasets
            datasets_list = []
            for source in self.sources:
                ds = self._load_single_source(source)
                datasets_list.append(ds)
            
            # Calculate normalized probabilities
            weights = [s.weight for s in self.sources]
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]
            
            logger.info(f"Sampling probabilities: {probabilities}")
            
            # Mix datasets with interleave_datasets
            # stopping_strategy="all_exhausted" ensures all datasets
            # are used completely
            interleaved_dataset = interleave_datasets(
                datasets_list,
                probabilities=probabilities,
                seed=self.seed,
                stopping_strategy="first_exhausted"  # Stops when smallest finishes
            )
            
            logger.info(f"Mixed dataset: {len(interleaved_dataset)} total examples")
            
            # Split train/val
            split_dataset = interleaved_dataset.train_test_split(
                test_size=self.val_split_percentage,
                seed=self.seed,
            )
            
            train_data = split_dataset["train"]
            val_data = split_dataset["test"]
            
            # Tokenize
            self.train_dataset = train_data.map(
                self._tokenize_function,
                batched=True,
                remove_columns=train_data.column_names,
                desc="Tokenizing training",
            )
            
            self.val_dataset = val_data.map(
                self._tokenize_function,
                batched=True,
                remove_columns=val_data.column_names,
                desc="Tokenizing validation",
            )
            
            # Set PyTorch format for tensors
            self.train_dataset.set_format("torch")
            self.val_dataset.set_format("torch")
            
            logger.info("=" * 60)
            logger.info(
                f"Final dataset - Training: {len(self.train_dataset)} examples, "
                f"Validation: {len(self.val_dataset)} examples"
            )
            logger.info("=" * 60)
    
    def _tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize texts already formatted in ChatML.
        """
        texts = examples["text"]
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        
        # Create labels for causal LM
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
        """Create the DataLoader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Create the DataLoader for validation."""
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
# LEGACY SINGLE-SOURCE DATA MODULE (Backward Compatible)
# =============================================================================

class InstructionDataModule(pl.LightningDataModule):
    """
    LightningDataModule for instruction tuning datasets (single-source).
    
    Supports datasets from HuggingFace with instruction-response format
    (e.g., databricks/databricks-dolly-15k, alpaca, etc.).
    
    NOTE: For multi-source training, use MultiSourceDataModule.
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
        use_chatml: bool = True,  # New parameter for ChatML format
    ):
        """
        Initialize the DataModule.
        
        Args:
            tokenizer: Tokenizer for text tokenization
            dataset_name: Dataset name on HuggingFace or local path
            dataset_config: Specific dataset configuration (optional)
            text_column: Name of column with instructions/prompts
            response_column: Name of column with responses
            max_seq_length: Maximum length of tokenized sequences
            train_split: Name of training split
            val_split: Name of validation split (None = create from train)
            val_split_percentage: Percentage of train to use as val (if val_split=None)
            per_device_train_batch_size: Batch size per GPU for training
            per_device_eval_batch_size: Batch size per GPU for validation
            gradient_accumulation_steps: Gradient accumulation steps
            num_workers: Number of workers for DataLoader
            use_chatml: If True, use ChatML format instead of Alpaca
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
        Download and prepare the dataset (called only on rank 0 in DDP).
        Should not modify state (self).
        """
        logger.info(f"Preparing dataset: {self.dataset_name}")
        
        # Load the dataset (download only, no tokenization)
        if not self._is_local_path(self.dataset_name):
            load_dataset(
                self.dataset_name,
                name=self.dataset_config,
                split=None,
            )
        logger.info("Dataset downloaded and ready")
    
    def _is_local_path(self, path: str) -> bool:
        """Check if the path is a local file."""
        return Path(path).exists() or path.endswith(('.json', '.jsonl', '.csv', '.parquet'))
    
    def _load_local_dataset(self, file_path: str) -> DatasetDict:
        """Load a local dataset from JSON, JSONL, CSV, or Parquet file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
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
                f"Unsupported file format: {suffix}. "
                "Supported formats: .json, .jsonl, .csv, .parquet"
            )
        
        if isinstance(dataset, Dataset):
            dataset = DatasetDict({"train": dataset})
        elif "train" not in dataset:
            first_key = list(dataset.keys())[0]
            dataset = DatasetDict({"train": dataset[first_key]})
        
        return dataset
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Load and tokenize the dataset for training/validation."""
        if stage == "fit" or stage is None:
            logger.info("Setting up dataset for training")
            
            if self._is_local_path(self.dataset_name):
                logger.info(f"Loading local dataset: {self.dataset_name}")
                dataset_dict = self._load_local_dataset(self.dataset_name)
            else:
                logger.info(f"Loading dataset from HuggingFace: {self.dataset_name}")
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
                    f"Split '{self.train_split}' not found in dataset. "
                    f"Available splits: {list(dataset_dict.keys())}"
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
                    f"Created validation split: {len(val_data)} examples "
                    f"({self.val_split_percentage*100:.1f}% of training)"
                )
            else:
                val_data = None
            
            self.train_dataset = train_data.map(
                self._tokenize_function,
                batched=True,
                remove_columns=train_data.column_names,
                desc="Tokenizing training",
            )
            
            if val_data is not None:
                self.val_dataset = val_data.map(
                    self._tokenize_function,
                    batched=True,
                    remove_columns=val_data.column_names,
                    desc="Tokenizing validation",
                )
            else:
                self.val_dataset = None
            
            logger.info(
                f"Dataset prepared - Training: {len(self.train_dataset)} examples, "
                f"Validation: {len(self.val_dataset) if self.val_dataset else 0} examples"
            )
    
    def _tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize dataset examples."""
        instructions = examples[self.text_column]
        responses = examples[self.response_column]
        
        texts = []
        for instruction, response in zip(instructions, responses):
            if self.use_chatml:
                # Use ChatML format
                text = ChatMLFormatter.format_instruction_response(
                    instruction=str(instruction),
                    response=str(response)
                )
            else:
                # Legacy Alpaca format
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
        """Create the DataLoader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Create the DataLoader for validation."""
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
    Factory function to create the appropriate DataModule based on config.
    
    Args:
        tokenizer: Tokenizer for tokenization
        config: Complete configuration (from config.yaml)
        
    Returns:
        InstructionDataModule or MultiSourceDataModule based on config
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
            raise ValueError("Multi-source enabled but no datasets specified in 'sources'")
        
        training_config = config.get("training", {})
        data_config = config.get("data", {})
        
        logger.info(f"Creating MultiSourceDataModule with {len(sources)} datasets")
        
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
        
        logger.info("Creating InstructionDataModule (single-source)")
        
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
            use_chatml=True,  # Use ChatML by default for consistency
        )
