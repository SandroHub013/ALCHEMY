"""
Modulo per il caricamento di modelli LLM con quantizzazione 4-bit (QLoRA) e PEFT.

Questo modulo gestisce il caricamento di modelli base (Mistral, Llama, GPT-NeoX)
con ottimizzazioni per memoria:
- Quantizzazione 4-bit NF4 tramite bitsandbytes
- PEFT/LoRA per fine-tuning efficiente
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
    Classe per caricare modelli LLM con quantizzazione 4-bit e configurazione LoRA.
    
    Supporta modelli come:
    - Mistral 7B
    - Llama 2/3
    - GPT-NeoX 20B
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
        Inizializza il ModelLoader.
        
        Args:
            model_name_or_path: Path o nome del modello su HuggingFace
            quantization_config: Configurazione per quantizzazione 4-bit
            lora_config: Configurazione per LoRA (PEFT)
            trust_remote_code: Se True, permette l'esecuzione di codice remoto
            use_flash_attention_2: Se True, usa Flash Attention 2 (richiede installazione)
        """
        self.model_name_or_path = model_name_or_path
        self.trust_remote_code = trust_remote_code
        self.use_flash_attention_2 = use_flash_attention_2
        
        # Configurazione quantizzazione di default (4-bit NF4)
        self.quantization_config = quantization_config or {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
        
        # Configurazione LoRA di default
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
        Restituisce i moduli target per LoRA in base al tipo di modello.
        
        Args:
            model_name: Nome del modello (es. "mistral", "llama", "gpt-neox")
            
        Returns:
            Lista di nomi dei moduli da targetizzare con LoRA
        """
        model_lower = model_name.lower()
        
        if "mistral" in model_lower or "llama" in model_lower:
            # Mistral e Llama hanno architettura simile (Transformer decoder)
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
            # GPT-NeoX e Pythia usano nomi diversi
            return [
                "query_key_value",  # Equivalente a q_proj, k_proj, v_proj combinati
                "dense",  # Equivalente a o_proj
                "dense_h_to_4h",  # Equivalente a gate_proj/up_proj
                "dense_4h_to_h",  # Equivalente a down_proj
            ]
        elif "gpt2" in model_lower or "gpt-j" in model_lower:
            return ["c_attn", "c_proj", "c_fc"]
        else:
            # Default: prova con i moduli più comuni
            logger.warning(
                f"Modello {model_name} non riconosciuto. "
                "Usando moduli target di default. "
                "Potrebbe essere necessario specificare manualmente target_modules."
            )
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    def _create_bitsandbytes_config(self) -> Optional[BitsAndBytesConfig]:
        """Crea la configurazione bitsandbytes per quantizzazione 4-bit."""
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
        Carica il tokenizer per il modello.
        
        Returns:
            Tokenizer configurato
        """
        logger.info(f"Caricamento tokenizer da {self.model_name_or_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            padding_side="right",  # Importante per causal LM
        )
        
        # Imposta pad_token se non esiste (necessario per alcuni modelli)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info(f"Tokenizer caricato. Vocab size: {len(self.tokenizer)}")
        return self.tokenizer
    
    def load_model(self, enable_gradient_checkpointing: bool = True) -> torch.nn.Module:
        """
        Carica il modello con quantizzazione 4-bit e applica LoRA.
        
        Args:
            enable_gradient_checkpointing: Se True, abilita gradient checkpointing
            
        Returns:
            Modello configurato con PEFT/LoRA
        """
        logger.info(f"Caricamento modello {self.model_name_or_path} con QLoRA")
        
        # Determina i moduli target se non specificati
        if "target_modules" not in self.lora_config or not self.lora_config["target_modules"]:
            target_modules = self._get_target_modules_for_model(self.model_name_or_path)
            self.lora_config["target_modules"] = target_modules
            logger.info(f"Moduli target LoRA auto-rilevati: {target_modules}")
        
        # Configurazione bitsandbytes
        bnb_config = self._create_bitsandbytes_config()
        
        # Carica il modello base con quantizzazione
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",  # Distribuisce automaticamente su GPU disponibili
            trust_remote_code=self.trust_remote_code,
            torch_dtype=torch.float16,
            use_flash_attention_2=self.use_flash_attention_2,
        )
        
        # Prepara il modello per training k-bit
        if bnb_config is not None:
            model = prepare_model_for_kbit_training(model)
            logger.info("Modello preparato per training k-bit")
        
        # Abilita gradient checkpointing per risparmiare memoria
        if enable_gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing abilitato")
            else:
                logger.warning("Gradient checkpointing non supportato da questo modello")
        
        # Crea configurazione LoRA
        lora_config_obj = LoraConfig(**self.lora_config)
        
        # Applica LoRA al modello
        model = get_peft_model(model, lora_config_obj)
        
        # Log informazioni sul modello
        model.print_trainable_parameters()
        
        self.model = model
        logger.info("Modello caricato e configurato con LoRA")
        
        return model
    
    def get_model_and_tokenizer(
        self, enable_gradient_checkpointing: bool = True
    ) -> tuple[torch.nn.Module, AutoTokenizer]:
        """
        Carica sia il modello che il tokenizer.
        
        Args:
            enable_gradient_checkpointing: Se True, abilita gradient checkpointing
            
        Returns:
            Tupla (modello, tokenizer)
        """
        if self.tokenizer is None:
            self.load_tokenizer()
        
        if self.model is None:
            self.load_model(enable_gradient_checkpointing=enable_gradient_checkpointing)
        
        return self.model, self.tokenizer

