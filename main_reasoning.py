"""
Entry point per training avanzato con LUFFY e Search-R1.

Questo script permette di:
1. Addestrare modelli con LUFFY (off-policy learning da DeepSeek-R1)
2. Addestrare modelli con Search-R1 (reasoning con ricerca integrata)
3. Combinare entrambi per capacità di ragionamento avanzate

Uso:
    # Training con LUFFY
    python main_reasoning.py --mode luffy --config config/config.yaml

    # Training con Search-R1
    python main_reasoning.py --mode search-r1 --config config/config.yaml
    
    # Training combinato
    python main_reasoning.py --mode combined --config config/config.yaml

Riferimenti:
- LUFFY: https://arxiv.org/abs/2504.14945
- Search-R1: https://github.com/PeterGriffinJin/Search-R1
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import torch

from src.models import ModelLoader
from src.data import create_data_module
from src.reasoning import (
    LuffyTrainer,
    LuffyConfig,
    OffPolicyMode,
    create_luffy_trainer,
    SearchR1Trainer,
    SearchR1Config,
    SearchEngineType,
    create_search_r1_trainer,
    create_search_engine,
)

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Carica configurazione da file YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_reasoning_traces(traces_path: str) -> List[Dict[str, Any]]:
    """
    Carica tracce di ragionamento off-policy.
    
    Args:
        traces_path: Path al file JSON con le tracce
        
    Returns:
        Lista di tracce
    """
    import json
    
    if not os.path.exists(traces_path):
        logger.warning(f"File tracce non trovato: {traces_path}")
        return []
    
    with open(traces_path, "r", encoding="utf-8") as f:
        traces = json.load(f)
    
    logger.info(f"Caricate {len(traces)} tracce da {traces_path}")
    return traces


def load_knowledge_base(kb_path: str) -> List[str]:
    """
    Carica knowledge base per Search-R1.
    
    Args:
        kb_path: Path alla directory o file con i documenti
        
    Returns:
        Lista di documenti
    """
    documents = []
    kb_path = Path(kb_path)
    
    if kb_path.is_file():
        # Single file
        with open(kb_path, "r", encoding="utf-8") as f:
            if kb_path.suffix == ".json":
                import json
                data = json.load(f)
                documents = [d.get("content", str(d)) for d in data]
            else:
                documents = [f.read()]
    elif kb_path.is_dir():
        # Directory di file
        for file_path in kb_path.glob("**/*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append(f.read())
        for file_path in kb_path.glob("**/*.md"):
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append(f.read())
    
    logger.info(f"Caricati {len(documents)} documenti dalla knowledge base")
    return documents


def train_luffy(
    model,
    tokenizer,
    config: Dict[str, Any],
    train_dataset,
) -> Dict[str, Any]:
    """
    Training con LUFFY.
    
    Args:
        model: Modello con LoRA
        tokenizer: Tokenizer
        config: Configurazione completa
        train_dataset: Dataset di training
        
    Returns:
        Metriche di training
    """
    logger.info("=" * 60)
    logger.info("TRAINING CON LUFFY - Off-Policy Reasoning Learning")
    logger.info("=" * 60)
    
    luffy_config = config.get("luffy", {})
    
    # Crea trainer
    trainer = create_luffy_trainer(model, tokenizer, config)
    
    # Carica tracce off-policy se disponibili
    traces_path = luffy_config.get("off_policy_traces_path")
    if traces_path and os.path.exists(traces_path):
        num_traces = trainer.load_off_policy_traces(traces_path)
        logger.info(f"Caricate {num_traces} tracce off-policy")
    else:
        logger.warning(
            "Nessuna traccia off-policy trovata. "
            "LUFFY funzionerà in modalità ExGRPO (learning from own experience)"
        )
    
    # Training
    train_config = config.get("training", {})
    results = trainer.train(
        train_dataset=train_dataset,
        num_epochs=train_config.get("num_epochs", 3),
        batch_size=train_config.get("per_device_train_batch_size", 2),
        learning_rate=train_config.get("learning_rate", 2e-5),
        output_dir=train_config.get("output_dir", "./checkpoints/luffy"),
    )
    
    logger.info("Training LUFFY completato!")
    return results


def train_search_r1(
    model,
    tokenizer,
    config: Dict[str, Any],
    train_data: List[Dict[str, str]],
    knowledge_base: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Training con Search-R1.
    
    Args:
        model: Modello con LoRA
        tokenizer: Tokenizer
        config: Configurazione completa
        train_data: Dati di training [{"question": ..., "answer": ...}]
        knowledge_base: Documenti per search engine
        
    Returns:
        Metriche di training
    """
    logger.info("=" * 60)
    logger.info("TRAINING CON SEARCH-R1 - Reasoning with Search")
    logger.info("=" * 60)
    
    search_config = config.get("search_r1", {})
    
    # Crea search engine
    engine_type = search_config.get("search_engine_type", "hybrid")
    search_engine = create_search_engine(
        engine_type,
        documents=knowledge_base or [],
    )
    
    logger.info(f"Search engine: {engine_type}")
    if knowledge_base:
        logger.info(f"Knowledge base: {len(knowledge_base)} documenti")
    
    # Crea trainer
    trainer = create_search_r1_trainer(
        model, tokenizer, config, 
        documents=knowledge_base
    )
    
    # Training
    train_config = config.get("training", {})
    results = trainer.train(
        train_data=train_data,
        num_epochs=train_config.get("num_epochs", 3),
        batch_size=train_config.get("per_device_train_batch_size", 2),
        learning_rate=train_config.get("learning_rate", 2e-5),
        output_dir=train_config.get("output_dir", "./checkpoints/search_r1"),
    )
    
    logger.info("Training Search-R1 completato!")
    return results


def train_combined(
    model,
    tokenizer,
    config: Dict[str, Any],
    train_dataset,
    knowledge_base: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Training combinato LUFFY + Search-R1.
    
    Prima addestra con LUFFY per migliorare reasoning,
    poi fine-tuna con Search-R1 per integrare ricerca.
    
    Args:
        model: Modello con LoRA
        tokenizer: Tokenizer
        config: Configurazione completa
        train_dataset: Dataset di training
        knowledge_base: Documenti per search
        
    Returns:
        Metriche combinate
    """
    logger.info("=" * 60)
    logger.info("TRAINING COMBINATO - LUFFY + Search-R1")
    logger.info("=" * 60)
    
    results = {"luffy": None, "search_r1": None}
    
    # Fase 1: LUFFY
    logger.info("\n[Fase 1/2] Training LUFFY...")
    luffy_results = train_luffy(model, tokenizer, config, train_dataset)
    results["luffy"] = luffy_results
    
    # Fase 2: Search-R1
    logger.info("\n[Fase 2/2] Training Search-R1...")
    
    # Prepara dati per Search-R1
    train_data = []
    for item in train_dataset:
        if isinstance(item, dict):
            train_data.append({
                "question": item.get("prompt", item.get("instruction", "")),
                "answer": item.get("response", item.get("output", "")),
            })
    
    search_results = train_search_r1(
        model, tokenizer, config, train_data, knowledge_base
    )
    results["search_r1"] = search_results
    
    logger.info("\nTraining combinato completato!")
    return results


def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(
        description="Training avanzato con LUFFY e Search-R1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python main_reasoning.py --mode luffy --config config/config.yaml
  python main_reasoning.py --mode search-r1 --config config/config.yaml --kb ./data/knowledge_base
  python main_reasoning.py --mode combined --config config/config.yaml
        """
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["luffy", "search-r1", "combined"],
        default="luffy",
        help="Modalità di training (default: luffy)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path al file di configurazione",
    )
    parser.add_argument(
        "--traces",
        type=str,
        default=None,
        help="Path alle tracce off-policy (override config)",
    )
    parser.add_argument(
        "--kb",
        type=str,
        default=None,
        help="Path alla knowledge base per Search-R1",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory per checkpoint (override config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verifica configurazione senza training",
    )
    args = parser.parse_args()
    
    # Carica configurazione
    logger.info(f"Caricamento configurazione da {args.config}")
    config = load_config(args.config)
    
    # Override da CLI
    if args.traces:
        config.setdefault("luffy", {})["off_policy_traces_path"] = args.traces
    if args.output_dir:
        config.setdefault("training", {})["output_dir"] = args.output_dir
    
    # Crea directory output
    output_dir = config.get("training", {}).get("output_dir", "./checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurazione hardware
    hardware_config = config.get("hardware", {})
    num_gpus = hardware_config.get("num_gpus", 1)
    if num_gpus == -1:
        num_gpus = torch.cuda.device_count()
    
    logger.info(f"GPU disponibili: {torch.cuda.device_count()}, GPU da usare: {num_gpus}")
    
    # Carica modello
    logger.info("Caricamento modello e tokenizer...")
    model_config = config["model"]
    peft_config = config["peft"]
    
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
        enable_gradient_checkpointing=config.get("training", {}).get("gradient_checkpointing", True)
    )
    
    # Dry run
    if args.dry_run:
        logger.info("\n" + "=" * 60)
        logger.info("DRY RUN - Configurazione verificata")
        logger.info("=" * 60)
        logger.info(f"Modalità: {args.mode}")
        logger.info(f"Modello: {model_config['name_or_path']}")
        logger.info(f"Output: {output_dir}")
        
        if args.mode in ["luffy", "combined"]:
            luffy_cfg = config.get("luffy", {})
            logger.info(f"\nLUFFY Config:")
            logger.info(f"  Mode: {luffy_cfg.get('mode', 'luffy')}")
            logger.info(f"  Off-policy source: {luffy_cfg.get('off_policy_source', 'deepseek-r1')}")
            logger.info(f"  Traces path: {luffy_cfg.get('off_policy_traces_path', 'N/A')}")
        
        if args.mode in ["search-r1", "combined"]:
            search_cfg = config.get("search_r1", {})
            logger.info(f"\nSearch-R1 Config:")
            logger.info(f"  Engine: {search_cfg.get('search_engine_type', 'hybrid')}")
            logger.info(f"  Max searches: {search_cfg.get('max_search_calls', 3)}")
            logger.info(f"  Knowledge base: {args.kb or 'N/A'}")
        
        logger.info("\nConfigurazione OK! Rimuovi --dry-run per avviare il training.")
        return
    
    # Prepara dataset
    logger.info("Preparazione dataset...")
    data_module = create_data_module(tokenizer=tokenizer, config=config)
    data_module.setup()
    train_dataset = data_module.train_dataset
    
    # Carica knowledge base se specificata
    knowledge_base = None
    if args.kb:
        knowledge_base = load_knowledge_base(args.kb)
    
    # Training
    if args.mode == "luffy":
        results = train_luffy(model, tokenizer, config, train_dataset)
    elif args.mode == "search-r1":
        # Converti dataset per Search-R1
        train_data = []
        for item in train_dataset:
            if isinstance(item, dict):
                train_data.append({
                    "question": item.get("prompt", item.get("instruction", "")),
                    "answer": item.get("response", item.get("output", "")),
                })
        results = train_search_r1(model, tokenizer, config, train_data, knowledge_base)
    else:  # combined
        results = train_combined(model, tokenizer, config, train_dataset, knowledge_base)
    
    # Report finale
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETATO")
    logger.info("=" * 60)
    logger.info(f"Checkpoint salvati in: {output_dir}")
    
    if isinstance(results, dict):
        if "total_steps" in results:
            logger.info(f"Total steps: {results['total_steps']}")
        if "final_loss" in results:
            logger.info(f"Final loss: {results['final_loss']:.4f}")
        if "final_avg_reward" in results:
            logger.info(f"Final avg reward: {results['final_avg_reward']:.4f}")


if __name__ == "__main__":
    main()

