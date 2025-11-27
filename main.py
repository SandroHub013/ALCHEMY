"""
Entry point principale per il fine-tuning di LLM con Agent Lightning.

Questo script:
1. Carica la configurazione da config.yaml
2. Inizializza il modello con QLoRA
3. Prepara il dataset
4. Configura il trainer Lightning
5. Avvia il training

Uso:
    python main.py --config config/config.yaml
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.models import ModelLoader
from src.data import create_data_module
from src.agent import LLMTrainingAgent

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Carica la configurazione da file YAML.
    
    Args:
        config_path: Path al file di configurazione
        
    Returns:
        Dizionario con la configurazione
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def setup_loggers(config: Dict[str, Any]) -> list:
    """
    Configura i logger per Lightning (TensorBoard, WandB).
    
    Args:
        config: Configurazione completa
        
    Returns:
        Lista di logger per Lightning
    """
    loggers = []
    log_config = config.get("logging", {})
    
    # TensorBoard
    if log_config.get("use_tensorboard", True):
        log_dir = log_config.get("log_dir", "./logs")
        tb_logger = TensorBoardLogger(
            save_dir=log_dir,
            name="llm_finetuning",
        )
        loggers.append(tb_logger)
        logger.info(f"TensorBoard logging abilitato: {tb_logger.log_dir}")
    
    # WandB
    if log_config.get("use_wandb", False):
        wandb_project = log_config.get("wandb_project", "llm-finetuning")
        wandb_entity = log_config.get("wandb_entity", None)
        wandb_logger = WandbLogger(
            project=wandb_project,
            entity=wandb_entity,
        )
        loggers.append(wandb_logger)
        logger.info(f"WandB logging abilitato: {wandb_project}")
    
    return loggers


def create_callbacks(config: Dict[str, Any]) -> list:
    """
    Crea i callback per Lightning (checkpointing, LR monitoring).
    
    Args:
        config: Configurazione completa
        
    Returns:
        Lista di callback per Lightning
    """
    callbacks = []
    train_config = config.get("training", {})
    output_dir = train_config.get("output_dir", "./checkpoints")
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="checkpoint-{epoch:02d}-{val/loss:.3f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=1 if train_config.get("save_strategy") == "epoch" else None,
        every_n_train_steps=train_config.get("save_steps") if train_config.get("save_strategy") == "steps" else None,
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    return callbacks


def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(description="Fine-tuning LLM con Agent Lightning")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path al file di configurazione",
    )
    args = parser.parse_args()
    
    # Carica configurazione
    logger.info(f"Caricamento configurazione da {args.config}")
    config = load_config(args.config)
    
    # Crea directory di output
    output_dir = config["training"].get("output_dir", "./checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurazione hardware
    hardware_config = config.get("hardware", {})
    num_gpus = hardware_config.get("num_gpus", 1)
    if num_gpus == -1:
        num_gpus = torch.cuda.device_count()
    
    logger.info(f"GPU disponibili: {torch.cuda.device_count()}, GPU da usare: {num_gpus}")
    
    # Carica modello e tokenizer
    logger.info("Caricamento modello e tokenizer...")
    model_config = config["model"]
    peft_config = config["peft"]
    
    # Prepara configurazione quantizzazione
    quantization_config = peft_config.get("quantization", {})
    # Converti string dtype in torch dtype
    if isinstance(quantization_config.get("bnb_4bit_compute_dtype"), str):
        dtype_str = quantization_config["bnb_4bit_compute_dtype"]
        quantization_config["bnb_4bit_compute_dtype"] = getattr(torch, dtype_str, torch.float16)
    
    # Prepara configurazione LoRA
    lora_config = peft_config.get("lora", {})
    
    model_loader = ModelLoader(
        model_name_or_path=model_config["name_or_path"],
        quantization_config=quantization_config,
        lora_config=lora_config,
        trust_remote_code=model_config.get("trust_remote_code", False),
        use_flash_attention_2=model_config.get("use_flash_attention_2", False),
    )
    
    model, tokenizer = model_loader.get_model_and_tokenizer(
        enable_gradient_checkpointing=config["training"].get("gradient_checkpointing", True)
    )
    
    # Prepara dataset
    # Usa la factory function che sceglie automaticamente tra:
    # - MultiSourceDataModule (se datasets.multi_source_enabled = true)
    # - InstructionDataModule (legacy single-source)
    logger.info("Preparazione dataset...")
    
    datasets_config = config.get("datasets", {})
    if datasets_config.get("multi_source_enabled", False):
        logger.info("=" * 60)
        logger.info("MULTI-SOURCE TRAINING ABILITATO")
        logger.info("Questo crea un modello generalista (Code + Agent + Chat)")
        logger.info("=" * 60)
        sources = datasets_config.get("sources", [])
        for src in sources:
            logger.info(f"  - {src['name']}: {src.get('weight', 1.0)*100:.0f}% ({src.get('type', 'instruction')})")
    
    data_module = create_data_module(tokenizer=tokenizer, config=config)
    train_config = config["training"]
    
    # Crea training agent
    logger.info("Creazione training agent...")
    agent = LLMTrainingAgent(
        model=model,
        tokenizer=tokenizer,
        learning_rate=train_config.get("learning_rate", 2.0e-4),
        weight_decay=train_config.get("weight_decay", 0.01),
        warmup_steps=train_config.get("warmup_steps", 100),
        max_steps=None,  # Sarà calcolato da Lightning
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        max_grad_norm=train_config.get("max_grad_norm", 0.3),
        fp16=train_config.get("fp16", True),
        bf16=train_config.get("bf16", False),
    )
    
    # Configura logger e callback
    loggers = setup_loggers(config)
    callbacks = create_callbacks(config)
    
    # Configura precision
    precision = "16-mixed" if train_config.get("fp16", True) else "bf16-mixed" if train_config.get("bf16", False) else "32"
    
    # Crea trainer
    trainer = pl.Trainer(
        max_epochs=train_config.get("num_epochs", 3),
        accelerator="gpu" if num_gpus > 0 else "cpu",
        devices=num_gpus if num_gpus > 0 else 1,
        precision=precision,
        accumulate_grad_batches=train_config.get("gradient_accumulation_steps", 1),
        gradient_clip_val=train_config.get("max_grad_norm", 0.3),
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=train_config.get("logging_steps", 10),
        val_check_interval=1.0,  # Valida ogni epoch
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,  # Per performance (usa True per riproducibilità assoluta)
    )
    
    # Avvia training
    logger.info("Avvio training...")
    trainer.fit(agent, datamodule=data_module)
    
    logger.info("Training completato!")
    logger.info(f"Checkpoint salvati in: {output_dir}")


if __name__ == "__main__":
    main()

