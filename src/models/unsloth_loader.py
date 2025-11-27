"""
Unsloth Model Loader - High-performance model loading with 2x speed and 70% less VRAM.

This module provides an optimized model loader using Unsloth's FastLanguageModel
which offers significant improvements over standard HuggingFace loading:
- 2x faster training
- 70% less VRAM usage
- 10-13x longer context lengths
- Native RL support (GRPO, DPO, etc.)

References:
- GitHub: https://github.com/unslothai/unsloth
- Documentation: https://docs.unsloth.ai/
- License: Apache 2.0

Requirements:
    pip install unsloth
"""

from typing import Optional, Dict, Any, List, Tuple, Union, Literal
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Check if Unsloth is available
try:
    from unsloth import FastLanguageModel, FastModel
    from unsloth import is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    FastLanguageModel = None
    FastModel = None
    is_bfloat16_supported = lambda: False


class UnslothPrecision(str, Enum):
    """Supported precision modes in Unsloth."""
    FOUR_BIT = "4bit"      # QLoRA 4-bit quantization
    EIGHT_BIT = "8bit"     # 8-bit quantization
    SIXTEEN_BIT = "16bit"  # 16-bit LoRA (new in Unsloth)
    FULL = "full"          # Full precision fine-tuning


@dataclass
class UnslothConfig:
    """Configuration for Unsloth model loading."""
    
    # Model identification
    model_name: str = "unsloth/Mistral-7B-v0.3"
    
    # Precision mode
    precision: UnslothPrecision = UnslothPrecision.FOUR_BIT
    
    # Sequence length
    max_seq_length: int = 2048  # Unsloth supports very long contexts
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0  # 0 is optimized in Unsloth
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"  # "none" is optimized
    
    # Gradient checkpointing
    use_gradient_checkpointing: Union[bool, str] = "unsloth"  # "unsloth" saves 30% more VRAM
    
    # Advanced options
    use_rslora: bool = False    # Rank-stabilized LoRA
    loftq_config: Optional[Dict] = None  # LoftQ configuration
    random_state: int = 3407
    
    # Full fine-tuning (requires more VRAM)
    full_finetuning: bool = False
    
    # HuggingFace token for gated models
    hf_token: Optional[str] = None


# =============================================================================
# UNSLOTH MODEL LOADER
# =============================================================================

class UnslothModelLoader:
    """
    High-performance model loader using Unsloth.
    
    Unsloth provides significant optimizations:
    - 2x faster training speed
    - 70% less VRAM usage
    - Support for much longer context lengths
    - Native support for RL methods (GRPO, DPO, etc.)
    
    Example:
        ```python
        loader = UnslothModelLoader(
            model_name="mistralai/Mistral-7B-v0.3",
            precision="4bit",
            max_seq_length=4096,
        )
        
        model, tokenizer = loader.load()
        
        # Model is ready for training with TRL, PyTorch Lightning, etc.
        ```
    """
    
    # Pre-quantized models from Unsloth (4x faster download)
    PREQANTIZED_MODELS = {
        # Mistral
        "mistral-7b": "unsloth/mistral-7b-v0.3-bnb-4bit",
        "mistral-nemo": "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
        # Llama
        "llama-3.1-8b": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "llama-3.1-8b-instruct": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "llama-3.2-3b": "unsloth/Llama-3.2-3B-bnb-4bit",
        "llama-3.2-1b": "unsloth/Llama-3.2-1B-bnb-4bit",
        # Qwen
        "qwen-2.5-7b": "unsloth/Qwen2.5-7B-bnb-4bit",
        "qwen-2.5-7b-instruct": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "qwen-2.5-coder-7b": "unsloth/Qwen2.5-Coder-7B-bnb-4bit",
        # DeepSeek
        "deepseek-r1-8b": "unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit",
        "deepseek-coder-6.7b": "unsloth/deepseek-coder-6.7b-instruct-bnb-4bit",
        # Gemma
        "gemma-2-9b": "unsloth/gemma-2-9b-bnb-4bit",
        "gemma-2-2b": "unsloth/gemma-2-2b-bnb-4bit",
        # Phi
        "phi-3-mini": "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
        "phi-3.5-mini": "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
    }
    
    def __init__(
        self,
        model_name: str = "unsloth/Mistral-7B-v0.3",
        config: Optional[UnslothConfig] = None,
        **kwargs,
    ):
        """
        Initialize the Unsloth model loader.
        
        Args:
            model_name: Model name or path (HuggingFace or local)
            config: Unsloth configuration object
            **kwargs: Override config parameters
        """
        if not UNSLOTH_AVAILABLE:
            raise ImportError(
                "Unsloth is not installed. Install with:\n"
                "  pip install unsloth\n"
                "Or for the latest version:\n"
                "  pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'"
            )
        
        # Create or update config
        if config is None:
            config = UnslothConfig(model_name=model_name, **kwargs)
        else:
            config.model_name = model_name
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        self.config = config
        self.model = None
        self.tokenizer = None
        
        logger.info(f"UnslothModelLoader initialized for: {model_name}")
    
    @classmethod
    def get_prequantized_model(cls, alias: str) -> str:
        """
        Get a pre-quantized model name from alias.
        
        Pre-quantized models download 4x faster and prevent OOM errors.
        
        Args:
            alias: Short alias like "mistral-7b", "llama-3.1-8b", etc.
            
        Returns:
            Full model name from Unsloth hub
        """
        alias_lower = alias.lower()
        if alias_lower in cls.PREQANTIZED_MODELS:
            return cls.PREQANTIZED_MODELS[alias_lower]
        return alias
    
    def _determine_precision_flags(self) -> Dict[str, bool]:
        """Determine loading flags based on precision mode."""
        precision = self.config.precision
        
        if precision == UnslothPrecision.FOUR_BIT:
            return {
                "load_in_4bit": True,
                "load_in_8bit": False,
                "load_in_16bit": False,
                "full_finetuning": False,
            }
        elif precision == UnslothPrecision.EIGHT_BIT:
            return {
                "load_in_4bit": False,
                "load_in_8bit": True,
                "load_in_16bit": False,
                "full_finetuning": False,
            }
        elif precision == UnslothPrecision.SIXTEEN_BIT:
            return {
                "load_in_4bit": False,
                "load_in_8bit": False,
                "load_in_16bit": True,
                "full_finetuning": False,
            }
        else:  # FULL
            return {
                "load_in_4bit": False,
                "load_in_8bit": False,
                "load_in_16bit": False,
                "full_finetuning": True,
            }
    
    def load_base_model(self) -> Tuple[Any, Any]:
        """
        Load the base model and tokenizer with Unsloth optimizations.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model with Unsloth: {self.config.model_name}")
        logger.info(f"  Precision: {self.config.precision.value}")
        logger.info(f"  Max sequence length: {self.config.max_seq_length}")
        
        # Get precision flags
        precision_flags = self._determine_precision_flags()
        
        # Prepare kwargs
        load_kwargs = {
            "model_name": self.config.model_name,
            "max_seq_length": self.config.max_seq_length,
            **precision_flags,
        }
        
        # Add token if provided
        if self.config.hf_token:
            load_kwargs["token"] = self.config.hf_token
        
        # Load with FastLanguageModel (or FastModel for newer versions)
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
        except Exception as e:
            logger.warning(f"FastLanguageModel failed, trying FastModel: {e}")
            model, tokenizer = FastModel.from_pretrained(**load_kwargs)
        
        logger.info("Base model loaded successfully with Unsloth")
        
        return model, tokenizer
    
    def apply_peft(self, model: Any) -> Any:
        """
        Apply LoRA/PEFT to the model using Unsloth's optimized implementation.
        
        Unsloth's LoRA implementation uses 30% less VRAM than standard PEFT.
        
        Args:
            model: Base model from load_base_model()
            
        Returns:
            Model with LoRA adapters applied
        """
        logger.info("Applying Unsloth LoRA configuration...")
        logger.info(f"  LoRA r: {self.config.lora_r}")
        logger.info(f"  LoRA alpha: {self.config.lora_alpha}")
        logger.info(f"  Target modules: {self.config.target_modules}")
        logger.info(f"  Gradient checkpointing: {self.config.use_gradient_checkpointing}")
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.lora_r,
            target_modules=self.config.target_modules,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.bias,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            random_state=self.config.random_state,
            max_seq_length=self.config.max_seq_length,
            use_rslora=self.config.use_rslora,
            loftq_config=self.config.loftq_config,
        )
        
        logger.info("LoRA applied with Unsloth optimizations")
        
        return model
    
    def load(self, apply_lora: bool = True) -> Tuple[Any, Any]:
        """
        Load model and tokenizer with all optimizations.
        
        Args:
            apply_lora: Whether to apply LoRA (default True)
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Load base model
        model, tokenizer = self.load_base_model()
        
        # Apply LoRA if requested
        if apply_lora and not self.config.full_finetuning:
            model = self.apply_peft(model)
        
        self.model = model
        self.tokenizer = tokenizer
        
        # Print trainable parameters
        if hasattr(model, 'print_trainable_parameters'):
            model.print_trainable_parameters()
        
        return model, tokenizer
    
    def get_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """
        Get the loaded model and tokenizer (loads if not already loaded).
        
        Returns:
            Tuple of (model, tokenizer)
        """
        if self.model is None or self.tokenizer is None:
            return self.load()
        return self.model, self.tokenizer
    
    def enable_inference_mode(self) -> None:
        """
        Enable fast inference mode (2x faster generation).
        
        Call this before generation to enable Unsloth's inference optimizations.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        FastLanguageModel.for_inference(self.model)
        logger.info("Inference mode enabled (2x faster generation)")
    
    def enable_training_mode(self) -> None:
        """
        Enable training mode (after inference mode).
        
        Call this before training if you previously called enable_inference_mode().
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        FastLanguageModel.for_training(self.model)
        logger.info("Training mode enabled")
    
    def save_model(
        self,
        output_dir: str,
        save_method: Literal["lora", "merged_16bit", "merged_4bit", "gguf"] = "lora",
        quantization_method: Optional[str] = None,
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
    ) -> str:
        """
        Save the model using various methods.
        
        Args:
            output_dir: Directory to save the model
            save_method: How to save the model:
                - "lora": Save only LoRA adapters (smallest)
                - "merged_16bit": Merge LoRA and save in 16-bit
                - "merged_4bit": Merge LoRA and save in 4-bit
                - "gguf": Save as GGUF for llama.cpp
            quantization_method: For GGUF, quantization type (e.g., "q4_k_m")
            push_to_hub: Whether to push to HuggingFace Hub
            hub_model_id: Model ID for Hub upload
            
        Returns:
            Path to saved model
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        logger.info(f"Saving model with method: {save_method}")
        
        if save_method == "lora":
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
        elif save_method == "merged_16bit":
            self.model.save_pretrained_merged(
                output_dir,
                self.tokenizer,
                save_method="merged_16bit",
            )
            
        elif save_method == "merged_4bit":
            self.model.save_pretrained_merged(
                output_dir,
                self.tokenizer,
                save_method="merged_4bit",
            )
            
        elif save_method == "gguf":
            quant = quantization_method or "q4_k_m"
            self.model.save_pretrained_gguf(
                output_dir,
                self.tokenizer,
                quantization_method=quant,
            )
            logger.info(f"GGUF saved with quantization: {quant}")
        
        # Push to Hub if requested
        if push_to_hub and hub_model_id:
            if save_method == "gguf":
                self.model.push_to_hub_gguf(
                    hub_model_id,
                    self.tokenizer,
                    quantization_method=quantization_method or "q4_k_m",
                )
            else:
                self.model.push_to_hub(hub_model_id)
                self.tokenizer.push_to_hub(hub_model_id)
            logger.info(f"Pushed to Hub: {hub_model_id}")
        
        logger.info(f"Model saved to: {output_dir}")
        return output_dir


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_unsloth_loader(
    model_name: str,
    precision: str = "4bit",
    max_seq_length: int = 2048,
    lora_r: int = 16,
    **kwargs,
) -> UnslothModelLoader:
    """
    Factory function to create an UnslothModelLoader.
    
    Args:
        model_name: Model name or alias
        precision: "4bit", "8bit", "16bit", or "full"
        max_seq_length: Maximum sequence length
        lora_r: LoRA rank
        **kwargs: Additional configuration
        
    Returns:
        Configured UnslothModelLoader
    """
    # Convert alias to full name if available
    model_name = UnslothModelLoader.get_prequantized_model(model_name)
    
    # Convert precision string to enum
    precision_enum = UnslothPrecision(precision)
    
    config = UnslothConfig(
        model_name=model_name,
        precision=precision_enum,
        max_seq_length=max_seq_length,
        lora_r=lora_r,
        **kwargs,
    )
    
    return UnslothModelLoader(model_name=model_name, config=config)


def load_with_unsloth(
    model_name: str,
    precision: str = "4bit",
    max_seq_length: int = 2048,
    **kwargs,
) -> Tuple[Any, Any]:
    """
    Quick helper to load a model with Unsloth.
    
    Args:
        model_name: Model name or alias
        precision: Precision mode
        max_seq_length: Max sequence length
        **kwargs: Additional configuration
        
    Returns:
        Tuple of (model, tokenizer)
    """
    loader = create_unsloth_loader(
        model_name=model_name,
        precision=precision,
        max_seq_length=max_seq_length,
        **kwargs,
    )
    return loader.load()


def check_unsloth_available() -> bool:
    """Check if Unsloth is installed and available."""
    return UNSLOTH_AVAILABLE


def get_recommended_model(
    task: str = "general",
    vram_gb: int = 8,
) -> str:
    """
    Get recommended model based on task and available VRAM.
    
    Args:
        task: "general", "coding", "reasoning", "chat"
        vram_gb: Available VRAM in GB
        
    Returns:
        Recommended model name
    """
    recommendations = {
        "general": {
            8: "unsloth/Llama-3.2-3B-bnb-4bit",
            12: "unsloth/Mistral-7B-v0.3-bnb-4bit",
            16: "unsloth/Qwen2.5-7B-bnb-4bit",
            24: "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
            40: "unsloth/Qwen2.5-32B-bnb-4bit",
        },
        "coding": {
            8: "unsloth/Llama-3.2-3B-bnb-4bit",
            12: "unsloth/deepseek-coder-6.7b-instruct-bnb-4bit",
            16: "unsloth/Qwen2.5-Coder-7B-bnb-4bit",
            24: "unsloth/DeepSeek-Coder-V2-Lite-Instruct-bnb-4bit",
        },
        "reasoning": {
            8: "unsloth/Llama-3.2-3B-bnb-4bit",
            12: "unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit",
            16: "unsloth/Qwen2.5-7B-bnb-4bit",
            24: "unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit",
        },
        "chat": {
            8: "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
            12: "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit",
            16: "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
            24: "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        },
    }
    
    task_recs = recommendations.get(task, recommendations["general"])
    
    # Find the best model for available VRAM
    available_vrams = sorted(task_recs.keys())
    for vram in available_vrams:
        if vram >= vram_gb:
            return task_recs[vram]
    
    # Return the largest if VRAM exceeds all options
    return task_recs[max(available_vrams)]

