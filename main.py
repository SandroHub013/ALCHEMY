"""
Main entry point for LLM fine-tuning with Agent Lightning.

This script:
1. Loads configuration from config.yaml
2. Initializes the model with QLoRA
3. Prepares the dataset
4. Configures the Lightning trainer
5. Starts training

Usage:
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

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary with the configuration
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def setup_loggers(config: Dict[str, Any]) -> list:
    """
    Configure loggers for Lightning (TensorBoard, WandB).
    
    Args:
        config: Complete configuration
        
    Returns:
        List of loggers for Lightning
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
        logger.info(f"TensorBoard logging enabled: {tb_logger.log_dir}")
    
    # WandB
    if log_config.get("use_wandb", False):
        wandb_project = log_config.get("wandb_project", "llm-finetuning")
        wandb_entity = log_config.get("wandb_entity", None)
        wandb_logger = WandbLogger(
            project=wandb_project,
            entity=wandb_entity,
        )
        loggers.append(wandb_logger)
        logger.info(f"WandB logging enabled: {wandb_project}")
    
    return loggers


def create_callbacks(config: Dict[str, Any]) -> list:
    """
    Create callbacks for Lightning (checkpointing, LR monitoring).
    
    Args:
        config: Complete configuration
        
    Returns:
        List of callbacks for Lightning
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
    """Main function."""
    parser = argparse.ArgumentParser(description="LLM Fine-tuning with Agent Lightning")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Create output directory
    output_dir = config["training"].get("output_dir", "./checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    
    # Hardware configuration
    hardware_config = config.get("hardware", {})
    num_gpus = hardware_config.get("num_gpus", 1)
    if num_gpus == -1:
        num_gpus = torch.cuda.device_count()
    
    logger.info(f"GPUs available: {torch.cuda.device_count()}, GPUs to use: {num_gpus}")
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model_config = config["model"]
    peft_config = config["peft"]
    
    # Prepare quantization configuration
    quantization_config = peft_config.get("quantization", {})
    # Convert string dtype to torch dtype
    if isinstance(quantization_config.get("bnb_4bit_compute_dtype"), str):
        dtype_str = quantization_config["bnb_4bit_compute_dtype"]
        quantization_config["bnb_4bit_compute_dtype"] = getattr(torch, dtype_str, torch.float16)
    
    # Prepare LoRA configuration
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
    
    # Prepare dataset
    # Uses the factory function that automatically chooses between:
    # - MultiSourceDataModule (if datasets.multi_source_enabled = true)
    # - InstructionDataModule (legacy single-source)
    logger.info("Preparing dataset...")
    
    datasets_config = config.get("datasets", {})
    if datasets_config.get("multi_source_enabled", False):
        logger.info("=" * 60)
        logger.info("MULTI-SOURCE TRAINING ENABLED")
        logger.info("This creates a generalist model (Code + Agent + Chat)")
        logger.info("=" * 60)
        sources = datasets_config.get("sources", [])
        for src in sources:
            logger.info(f"  - {src['name']}: {src.get('weight', 1.0)*100:.0f}% ({src.get('type', 'instruction')})")
    
    data_module = create_data_module(tokenizer=tokenizer, config=config)
    train_config = config["training"]
    
    # Create training agent
    logger.info("Creating training agent...")
    agent = LLMTrainingAgent(
        model=model,
        tokenizer=tokenizer,
        learning_rate=train_config.get("learning_rate", 2.0e-4),
        weight_decay=train_config.get("weight_decay", 0.01),
        warmup_steps=train_config.get("warmup_steps", 100),
        max_steps=None,  # Will be calculated by Lightning
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        max_grad_norm=train_config.get("max_grad_norm", 0.3),
        fp16=train_config.get("fp16", True),
        bf16=train_config.get("bf16", False),
    )
    
    # Configure loggers and callbacks
    loggers = setup_loggers(config)
    callbacks = create_callbacks(config)
    
    # Configure precision
    precision = "16-mixed" if train_config.get("fp16", True) else "bf16-mixed" if train_config.get("bf16", False) else "32"
    
    # Create trainer
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
        val_check_interval=1.0,  # Validate every epoch
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,  # For performance (use True for absolute reproducibility)
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.fit(agent, datamodule=data_module)
    
    logger.info("Training completed!")
    logger.info(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
