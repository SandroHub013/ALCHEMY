"""
Training with Microsoft Agent Lightning - Reinforcement Learning for AI Agents.

This script uses Agent Lightning to train models with:
- GRPO (Group Relative Policy Optimization) - RL for agents
- APO (Automatic Prompt Optimization) - Prompt optimization
- Advanced SFT with tracing

Unlike main.py (classic PyTorch Lightning), this script
leverages Agent Lightning's RL algorithms to improve agent behavior
through custom reward functions.

Usage:
    python main_agent_lightning.py --config config/config.yaml

Prerequisites:
    pip install agentlightning

GitHub: https://github.com/microsoft/agent-lightning
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any
import yaml
import torch

from src.models import ModelLoader
from src.data import create_data_module
from src.agent import (
    AgentLightningTrainer, 
    AgentLightningConfig, 
    TrainingAlgorithm,
    RewardFunction,
    check_agent_lightning_available,
    create_agent_lightning_trainer,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def print_banner():
    """Print the Agent Lightning banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     ⚡ AGENT LIGHTNING - RL Training for AI Agents ⚡        ║
║                                                               ║
║     Microsoft Open Source                                     ║
║     https://github.com/microsoft/agent-lightning              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main function for training with Agent Lightning."""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="LLM Training with Agent Lightning (RL for agents)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["sft", "grpo", "apo"],
        default=None,
        help="Override algorithm (default: from config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only verify configuration without training",
    )
    args = parser.parse_args()
    
    # Verify Agent Lightning
    if not check_agent_lightning_available():
        logger.error(
            "❌ Agent Lightning not installed!\n"
            "   Install with: pip install agentlightning\n"
            "   Or use main.py for classic training."
        )
        return 1
    
    logger.info("✅ Agent Lightning available")
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override algorithm if specified
    if args.algorithm:
        config.setdefault("agent_lightning", {})["algorithm"] = args.algorithm
    
    # Verify Agent Lightning configuration
    agl_config = config.get("agent_lightning", {})
    if not agl_config.get("enabled", True):
        logger.warning(
            "Agent Lightning disabled in config. "
            "Set agent_lightning.enabled = true to use it."
        )
        return 1
    
    algorithm = agl_config.get("algorithm", "sft").upper()
    logger.info(f"Selected algorithm: {algorithm}")
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("AGENT LIGHTNING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"  Algorithm: {algorithm}")
    logger.info(f"  Reward Function: {agl_config.get('reward_function', 'combined')}")
    logger.info(f"  Tracing: {agl_config.get('enable_tracing', True)}")
    
    if algorithm == "GRPO":
        grpo_config = agl_config.get("grpo", {})
        logger.info(f"  GRPO Config:")
        logger.info(f"    - Generations per prompt: {grpo_config.get('num_generations', 4)}")
        logger.info(f"    - Temperature: {grpo_config.get('temperature', 0.7)}")
        logger.info(f"    - KL Coef: {grpo_config.get('kl_coef', 0.1)}")
    logger.info("=" * 60)
    
    if args.dry_run:
        logger.info("🔍 Dry-run completed. Configuration is valid.")
        return 0
    
    # Create output directory
    output_dir = config["training"].get("output_dir", "./checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    
    # Hardware configuration
    hardware_config = config.get("hardware", {})
    num_gpus = hardware_config.get("num_gpus", 1)
    if num_gpus == -1:
        num_gpus = torch.cuda.device_count()
    
    logger.info(f"GPUs available: {torch.cuda.device_count()}, GPUs to use: {num_gpus}")
    
    if num_gpus == 0:
        logger.warning("⚠️ No GPU detected. Training on CPU will be very slow!")
    
    # Load model and tokenizer
    logger.info("🔄 Loading model and tokenizer...")
    model_config = config["model"]
    peft_config = config["peft"]
    
    # Prepare quantization configuration
    quantization_config = peft_config.get("quantization", {})
    if isinstance(quantization_config.get("bnb_4bit_compute_dtype"), str):
        dtype_str = quantization_config["bnb_4bit_compute_dtype"]
        quantization_config["bnb_4bit_compute_dtype"] = getattr(torch, dtype_str, torch.float16)
    
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
    
    logger.info(f"✅ Model loaded: {model_config['name_or_path']}")
    
    # Prepare dataset
    logger.info("🔄 Preparing dataset...")
    
    datasets_config = config.get("datasets", {})
    if datasets_config.get("multi_source_enabled", False):
        logger.info("📊 Multi-Source Training enabled:")
        for src in datasets_config.get("sources", []):
            logger.info(f"   - {src['name']}: {src.get('weight', 1.0)*100:.0f}%")
    
    data_module = create_data_module(tokenizer=tokenizer, config=config)
    data_module.setup("fit")
    
    train_dataset = data_module.train_dataset
    eval_dataset = data_module.val_dataset
    
    logger.info(f"✅ Dataset prepared: {len(train_dataset)} training, {len(eval_dataset) if eval_dataset else 0} validation")
    
    # Create Agent Lightning Trainer
    logger.info("🔄 Creating Agent Lightning Trainer...")
    
    trainer = create_agent_lightning_trainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )
    
    logger.info("✅ Trainer created")
    
    # Training
    train_config = config["training"]
    
    logger.info("=" * 60)
    logger.info("🚀 STARTING TRAINING WITH AGENT LIGHTNING")
    logger.info("=" * 60)
    
    results = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=train_config.get("num_epochs", 3),
        batch_size=train_config.get("per_device_train_batch_size", 2),
        learning_rate=train_config.get("learning_rate", 2e-4),
        output_dir=output_dir,
    )
    
    # Log results
    logger.info("=" * 60)
    logger.info("✅ TRAINING COMPLETED!")
    logger.info("=" * 60)
    logger.info(f"Checkpoints saved to: {output_dir}")
    
    if results:
        logger.info("Final metrics:")
        for key, value in results.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    # Test generation
    logger.info("\n🧪 Test generation with trained model:")
    test_prompts = [
        "Write a Python function to calculate fibonacci numbers.",
        "What tools do you have available to help the user?",
        "Explain what machine learning is.",
    ]
    
    for prompt in test_prompts:
        logger.info(f"\n📝 Prompt: {prompt[:50]}...")
        try:
            response = trainer.generate(prompt, max_new_tokens=100)
            logger.info(f"🤖 Response: {response[:200]}...")
            
            # Evaluate reward
            reward = trainer.evaluate_reward(prompt, response)
            logger.info(f"⭐ Reward: {reward:.3f}")
        except Exception as e:
            logger.error(f"Generation error: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Agent Lightning pipeline completed successfully!")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
