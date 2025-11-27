"""
Module for loading LLM models with 4-bit quantization (QLoRA) and PEFT.

This module handles loading base models (Mistral, Llama, GPT-NeoX, DeepSeek)
with memory optimizations:
- 4-bit NF4 quantization via bitsandbytes
- PEFT/LoRA for efficient fine-tuning
- Gradient checkpointing
"""

from typing import Optional, Dict, Any, List
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Class for loading LLM models with 4-bit quantization and LoRA configuration.
    
    Supports models such as:
    - Mistral 7B
    - Llama 2/3
    - GPT-NeoX 20B
    - DeepSeek Qwen Distill
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        quantization_config: Optional[Dict[str, Any]] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        trust_remote_code: bool = False,
        use_flash_attention_2: bool = False,
    ):
        """
        Initialize the ModelLoader.
        
        Args:
            model_name_or_path: Path or name of the model on HuggingFace
            quantization_config: Configuration for 4-bit quantization
            lora_config: Configuration for LoRA (PEFT)
            trust_remote_code: If True, allows execution of remote code
            use_flash_attention_2: If True, uses Flash Attention 2 (requires installation)
        """
        self.model_name_or_path = model_name_or_path
        self.trust_remote_code = trust_remote_code
        self.use_flash_attention_2 = use_flash_attention_2
        
        # Default quantization configuration (4-bit NF4)
        self.quantization_config = quantization_config or {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
        
        # Default LoRA configuration
        self.lora_config = lora_config or {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        
        self.model = None
        self.tokenizer = None
    
    def _get_target_modules_for_model(self, model_name: str) -> List[str]:
        """
        Return target modules for LoRA based on model type.
        
        Args:
            model_name: Model name (e.g., "mistral", "llama", "gpt-neox")
            
        Returns:
            List of module names to target with LoRA
        """
        model_lower = model_name.lower()
        
        if "mistral" in model_lower or "llama" in model_lower or "qwen" in model_lower or "deepseek" in model_lower:
            # Mistral, Llama, Qwen, DeepSeek have similar architecture (Transformer decoder)
            return [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        elif "gpt-neox" in model_lower or "pythia" in model_lower:
            # GPT-NeoX and Pythia use different names
            return [
                "query_key_value",  # Equivalent to combined q_proj, k_proj, v_proj
                "dense",  # Equivalent to o_proj
                "dense_h_to_4h",  # Equivalent to gate_proj/up_proj
                "dense_4h_to_h",  # Equivalent to down_proj
            ]
        elif "gpt2" in model_lower or "gpt-j" in model_lower:
            return ["c_attn", "c_proj", "c_fc"]
        else:
            # Default: try with most common modules
            logger.warning(
                f"Model {model_name} not recognized. "
                "Using default target modules. "
                "You may need to manually specify target_modules."
            )
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    def _create_bitsandbytes_config(self) -> Optional[BitsAndBytesConfig]:
        """Create bitsandbytes configuration for 4-bit quantization."""
        if not self.quantization_config.get("load_in_4bit", False):
            return None
        
        compute_dtype_str = self.quantization_config.get("bnb_4bit_compute_dtype", "float16")
        if isinstance(compute_dtype_str, str):
            compute_dtype = getattr(torch, compute_dtype_str, torch.float16)
        else:
            compute_dtype = compute_dtype_str
        
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=self.quantization_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=self.quantization_config.get(
                "bnb_4bit_use_double_quant", True
            ),
        )
    
    def load_tokenizer(self) -> AutoTokenizer:
        """
        Load the tokenizer for the model.
        
        Returns:
            Configured tokenizer
        """
        logger.info(f"Loading tokenizer from {self.model_name_or_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            padding_side="right",  # Important for causal LM
        )
        
        # Set pad_token if it doesn't exist (required for some models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
        return self.tokenizer
    
    def load_model(self, enable_gradient_checkpointing: bool = True) -> torch.nn.Module:
        """
        Load the model with 4-bit quantization and apply LoRA.
        
        Args:
            enable_gradient_checkpointing: If True, enables gradient checkpointing
            
        Returns:
            Model configured with PEFT/LoRA
        """
        logger.info(f"Loading model {self.model_name_or_path} with QLoRA")
        
        # Determine target modules if not specified
        if "target_modules" not in self.lora_config or not self.lora_config["target_modules"]:
            target_modules = self._get_target_modules_for_model(self.model_name_or_path)
            self.lora_config["target_modules"] = target_modules
            logger.info(f"LoRA target modules auto-detected: {target_modules}")
        
        # bitsandbytes configuration
        bnb_config = self._create_bitsandbytes_config()
        
        # Load base model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",  # Automatically distribute across available GPUs
            trust_remote_code=self.trust_remote_code,
            torch_dtype=torch.float16,
            use_flash_attention_2=self.use_flash_attention_2,
        )
        
        # Prepare model for k-bit training
        if bnb_config is not None:
            model = prepare_model_for_kbit_training(model)
            logger.info("Model prepared for k-bit training")
        
        # Enable gradient checkpointing to save memory
        if enable_gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            else:
                logger.warning("Gradient checkpointing not supported by this model")
        
        # Create LoRA configuration
        lora_config_obj = LoraConfig(**self.lora_config)
        
        # Apply LoRA to the model
        model = get_peft_model(model, lora_config_obj)
        
        # Log model information
        model.print_trainable_parameters()
        
        self.model = model
        logger.info("Model loaded and configured with LoRA")
        
        return model
    
    def get_model_and_tokenizer(
        self, enable_gradient_checkpointing: bool = True
    ) -> tuple[torch.nn.Module, AutoTokenizer]:
        """
        Load both the model and tokenizer.
        
        Args:
            enable_gradient_checkpointing: If True, enables gradient checkpointing
            
        Returns:
            Tuple (model, tokenizer)
        """
        if self.tokenizer is None:
            self.load_tokenizer()
        
        if self.model is None:
            self.load_model(enable_gradient_checkpointing=enable_gradient_checkpointing)
        
        return self.model, self.tokenizer
