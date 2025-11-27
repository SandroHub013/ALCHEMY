"""
Training con Microsoft Agent Lightning - Reinforcement Learning per Agenti AI.

Questo script usa Agent Lightning per allenare modelli con:
- GRPO (Group Relative Policy Optimization) - RL per agenti
- APO (Automatic Prompt Optimization) - Ottimizzazione prompt
- SFT avanzato con tracciamento

A differenza di main.py (PyTorch Lightning classico), questo script
sfrutta gli algoritmi RL di Agent Lightning per migliorare il comportamento
dell'agente attraverso reward functions personalizzate.

Uso:
    python main_agent_lightning.py --config config/config.yaml

Prerequisiti:
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

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Carica la configurazione da file YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def print_banner():
    """Stampa il banner Agent Lightning."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     ⚡ AGENT LIGHTNING - Training RL per Agenti AI ⚡        ║
║                                                               ║
║     Microsoft Open Source                                     ║
║     https://github.com/microsoft/agent-lightning              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Funzione principale per training con Agent Lightning."""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="Training LLM con Agent Lightning (RL per agenti)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path al file di configurazione",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["sft", "grpo", "apo"],
        default=None,
        help="Override algoritmo (default: da config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo verifica configurazione senza training",
    )
    args = parser.parse_args()
    
    # Verifica Agent Lightning
    if not check_agent_lightning_available():
        logger.error(
            "❌ Agent Lightning non installato!\n"
            "   Installa con: pip install agentlightning\n"
            "   Oppure usa main.py per training classico."
        )
        return 1
    
    logger.info("✅ Agent Lightning disponibile")
    
    # Carica configurazione
    logger.info(f"Caricamento configurazione da {args.config}")
    config = load_config(args.config)
    
    # Override algoritmo se specificato
    if args.algorithm:
        config.setdefault("agent_lightning", {})["algorithm"] = args.algorithm
    
    # Verifica configurazione Agent Lightning
    agl_config = config.get("agent_lightning", {})
    if not agl_config.get("enabled", True):
        logger.warning(
            "Agent Lightning disabilitato nella config. "
            "Imposta agent_lightning.enabled = true per usarlo."
        )
        return 1
    
    algorithm = agl_config.get("algorithm", "sft").upper()
    logger.info(f"Algoritmo selezionato: {algorithm}")
    
    # Log configurazione
    logger.info("=" * 60)
    logger.info("CONFIGURAZIONE AGENT LIGHTNING")
    logger.info("=" * 60)
    logger.info(f"  Algoritmo: {algorithm}")
    logger.info(f"  Reward Function: {agl_config.get('reward_function', 'combined')}")
    logger.info(f"  Tracciamento: {agl_config.get('enable_tracing', True)}")
    
    if algorithm == "GRPO":
        grpo_config = agl_config.get("grpo", {})
        logger.info(f"  GRPO Config:")
        logger.info(f"    - Generazioni per prompt: {grpo_config.get('num_generations', 4)}")
        logger.info(f"    - Temperature: {grpo_config.get('temperature', 0.7)}")
        logger.info(f"    - KL Coef: {grpo_config.get('kl_coef', 0.1)}")
    logger.info("=" * 60)
    
    if args.dry_run:
        logger.info("🔍 Dry-run completato. Configurazione valida.")
        return 0
    
    # Crea directory di output
    output_dir = config["training"].get("output_dir", "./checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurazione hardware
    hardware_config = config.get("hardware", {})
    num_gpus = hardware_config.get("num_gpus", 1)
    if num_gpus == -1:
        num_gpus = torch.cuda.device_count()
    
    logger.info(f"GPU disponibili: {torch.cuda.device_count()}, GPU da usare: {num_gpus}")
    
    if num_gpus == 0:
        logger.warning("⚠️ Nessuna GPU rilevata. Training su CPU sarà molto lento!")
    
    # Carica modello e tokenizer
    logger.info("🔄 Caricamento modello e tokenizer...")
    model_config = config["model"]
    peft_config = config["peft"]
    
    # Prepara configurazione quantizzazione
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
    
    logger.info(f"✅ Modello caricato: {model_config['name_or_path']}")
    
    # Prepara dataset
    logger.info("🔄 Preparazione dataset...")
    
    datasets_config = config.get("datasets", {})
    if datasets_config.get("multi_source_enabled", False):
        logger.info("📊 Multi-Source Training abilitato:")
        for src in datasets_config.get("sources", []):
            logger.info(f"   - {src['name']}: {src.get('weight', 1.0)*100:.0f}%")
    
    data_module = create_data_module(tokenizer=tokenizer, config=config)
    data_module.setup("fit")
    
    train_dataset = data_module.train_dataset
    eval_dataset = data_module.val_dataset
    
    logger.info(f"✅ Dataset preparato: {len(train_dataset)} training, {len(eval_dataset) if eval_dataset else 0} validation")
    
    # Crea Agent Lightning Trainer
    logger.info("🔄 Creazione Agent Lightning Trainer...")
    
    trainer = create_agent_lightning_trainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )
    
    logger.info("✅ Trainer creato")
    
    # Training
    train_config = config["training"]
    
    logger.info("=" * 60)
    logger.info("🚀 AVVIO TRAINING CON AGENT LIGHTNING")
    logger.info("=" * 60)
    
    results = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=train_config.get("num_epochs", 3),
        batch_size=train_config.get("per_device_train_batch_size", 2),
        learning_rate=train_config.get("learning_rate", 2e-4),
        output_dir=output_dir,
    )
    
    # Log risultati
    logger.info("=" * 60)
    logger.info("✅ TRAINING COMPLETATO!")
    logger.info("=" * 60)
    logger.info(f"Checkpoint salvati in: {output_dir}")
    
    if results:
        logger.info("Metriche finali:")
        for key, value in results.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    # Test generazione
    logger.info("\n🧪 Test generazione con modello allenato:")
    test_prompts = [
        "Write a Python function to calculate fibonacci numbers.",
        "What tools do you have available to help the user?",
        "Spiega cos'è il machine learning in italiano.",
    ]
    
    for prompt in test_prompts:
        logger.info(f"\n📝 Prompt: {prompt[:50]}...")
        try:
            response = trainer.generate(prompt, max_new_tokens=100)
            logger.info(f"🤖 Response: {response[:200]}...")
            
            # Valuta reward
            reward = trainer.evaluate_reward(prompt, response)
            logger.info(f"⭐ Reward: {reward:.3f}")
        except Exception as e:
            logger.error(f"Errore generazione: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Pipeline Agent Lightning completata con successo!")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())

