"""
Entry point for advanced training with LUFFY and Search-R1.

This script allows you to:
1. Train models with LUFFY (off-policy learning from DeepSeek-R1)
2. Train models with Search-R1 (reasoning with integrated search)
3. Combine both for advanced reasoning capabilities

Usage:
    # Training with LUFFY
    python main_reasoning.py --mode luffy --config config/config.yaml

    # Training with Search-R1
    python main_reasoning.py --mode search-r1 --config config/config.yaml
    
    # Combined training
    python main_reasoning.py --mode combined --config config/config.yaml

References:
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


def load_reasoning_traces(traces_path: str) -> List[Dict[str, Any]]:
    """
    Load off-policy reasoning traces.
    
    Args:
        traces_path: Path to the JSON file with traces
        
    Returns:
        List of traces
    """
    import json
    
    if not os.path.exists(traces_path):
        logger.warning(f"Traces file not found: {traces_path}")
        return []
    
    with open(traces_path, "r", encoding="utf-8") as f:
        traces = json.load(f)
    
    logger.info(f"Loaded {len(traces)} traces from {traces_path}")
    return traces


def load_knowledge_base(kb_path: str) -> List[str]:
    """
    Load knowledge base for Search-R1.
    
    Args:
        kb_path: Path to the directory or file with documents
        
    Returns:
        List of documents
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
        # Directory of files
        for file_path in kb_path.glob("**/*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append(f.read())
        for file_path in kb_path.glob("**/*.md"):
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append(f.read())
    
    logger.info(f"Loaded {len(documents)} documents from knowledge base")
    return documents


def train_luffy(
    model,
    tokenizer,
    config: Dict[str, Any],
    train_dataset,
) -> Dict[str, Any]:
    """
    Training with LUFFY.
    
    Args:
        model: Model with LoRA
        tokenizer: Tokenizer
        config: Complete configuration
        train_dataset: Training dataset
        
    Returns:
        Training metrics
    """
    logger.info("=" * 60)
    logger.info("TRAINING WITH LUFFY - Off-Policy Reasoning Learning")
    logger.info("=" * 60)
    
    luffy_config = config.get("luffy", {})
    
    # Create trainer
    trainer = create_luffy_trainer(model, tokenizer, config)
    
    # Load off-policy traces if available
    traces_path = luffy_config.get("off_policy_traces_path")
    if traces_path and os.path.exists(traces_path):
        num_traces = trainer.load_off_policy_traces(traces_path)
        logger.info(f"Loaded {num_traces} off-policy traces")
    else:
        logger.warning(
            "No off-policy traces found. "
            "LUFFY will work in ExGRPO mode (learning from own experience)"
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
    
    logger.info("LUFFY training completed!")
    return results


def train_search_r1(
    model,
    tokenizer,
    config: Dict[str, Any],
    train_data: List[Dict[str, str]],
    knowledge_base: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Training with Search-R1.
    
    Args:
        model: Model with LoRA
        tokenizer: Tokenizer
        config: Complete configuration
        train_data: Training data [{"question": ..., "answer": ...}]
        knowledge_base: Documents for search engine
        
    Returns:
        Training metrics
    """
    logger.info("=" * 60)
    logger.info("TRAINING WITH SEARCH-R1 - Reasoning with Search")
    logger.info("=" * 60)
    
    search_config = config.get("search_r1", {})
    
    # Create search engine
    engine_type = search_config.get("search_engine_type", "hybrid")
    search_engine = create_search_engine(
        engine_type,
        documents=knowledge_base or [],
    )
    
    logger.info(f"Search engine: {engine_type}")
    if knowledge_base:
        logger.info(f"Knowledge base: {len(knowledge_base)} documents")
    
    # Create trainer
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
    
    logger.info("Search-R1 training completed!")
    return results


def train_combined(
    model,
    tokenizer,
    config: Dict[str, Any],
    train_dataset,
    knowledge_base: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Combined training LUFFY + Search-R1.
    
    First trains with LUFFY to improve reasoning,
    then fine-tunes with Search-R1 to integrate search.
    
    Args:
        model: Model with LoRA
        tokenizer: Tokenizer
        config: Complete configuration
        train_dataset: Training dataset
        knowledge_base: Documents for search
        
    Returns:
        Combined metrics
    """
    logger.info("=" * 60)
    logger.info("COMBINED TRAINING - LUFFY + Search-R1")
    logger.info("=" * 60)
    
    results = {"luffy": None, "search_r1": None}
    
    # Phase 1: LUFFY
    logger.info("\n[Phase 1/2] LUFFY Training...")
    luffy_results = train_luffy(model, tokenizer, config, train_dataset)
    results["luffy"] = luffy_results
    
    # Phase 2: Search-R1
    logger.info("\n[Phase 2/2] Search-R1 Training...")
    
    # Prepare data for Search-R1
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
    
    logger.info("\nCombined training completed!")
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Advanced training with LUFFY and Search-R1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
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
        help="Training mode (default: luffy)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--traces",
        type=str,
        default=None,
        help="Path to off-policy traces (override config)",
    )
    parser.add_argument(
        "--kb",
        type=str,
        default=None,
        help="Path to knowledge base for Search-R1",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for checkpoints (override config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify configuration without training",
    )
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override from CLI
    if args.traces:
        config.setdefault("luffy", {})["off_policy_traces_path"] = args.traces
    if args.output_dir:
        config.setdefault("training", {})["output_dir"] = args.output_dir
    
    # Create output directory
    output_dir = config.get("training", {}).get("output_dir", "./checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    
    # Hardware configuration
    hardware_config = config.get("hardware", {})
    num_gpus = hardware_config.get("num_gpus", 1)
    if num_gpus == -1:
        num_gpus = torch.cuda.device_count()
    
    logger.info(f"GPUs available: {torch.cuda.device_count()}, GPUs to use: {num_gpus}")
    
    # Load model
    logger.info("Loading model and tokenizer...")
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
        logger.info("DRY RUN - Configuration verified")
        logger.info("=" * 60)
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Model: {model_config['name_or_path']}")
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
        
        logger.info("\nConfiguration OK! Remove --dry-run to start training.")
        return
    
    # Prepare dataset
    logger.info("Preparing dataset...")
    data_module = create_data_module(tokenizer=tokenizer, config=config)
    data_module.setup()
    train_dataset = data_module.train_dataset
    
    # Load knowledge base if specified
    knowledge_base = None
    if args.kb:
        knowledge_base = load_knowledge_base(args.kb)
    
    # Training
    if args.mode == "luffy":
        results = train_luffy(model, tokenizer, config, train_dataset)
    elif args.mode == "search-r1":
        # Convert dataset for Search-R1
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
    
    # Final report
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Checkpoints saved to: {output_dir}")
    
    if isinstance(results, dict):
        if "total_steps" in results:
            logger.info(f"Total steps: {results['total_steps']}")
        if "final_loss" in results:
            logger.info(f"Final loss: {results['final_loss']:.4f}")
        if "final_avg_reward" in results:
            logger.info(f"Final avg reward: {results['final_avg_reward']:.4f}")


if __name__ == "__main__":
    main()
