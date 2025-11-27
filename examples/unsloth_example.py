"""
Example: Training with Unsloth for 2x Speed and 70% Less VRAM

This example demonstrates how to use Unsloth integration in ALCHEMY
for high-performance LLM fine-tuning and Reinforcement Learning.

Unsloth provides:
- 2x faster training
- 70% less VRAM usage
- 10-13x longer context lengths
- Native RL support (GRPO, DPO, etc.)

Reference: https://github.com/unslothai/unsloth

Requirements:
    pip install unsloth
    # Or for latest version:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

Run this example with:
    python examples/unsloth_example.py
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are available."""
    print("\n" + "="*60)
    print("Checking dependencies...")
    print("="*60 + "\n")
    
    # Check Unsloth
    try:
        from unsloth import FastLanguageModel
        print("✅ Unsloth is installed")
    except ImportError:
        print("❌ Unsloth not installed")
        print("   Install with: pip install unsloth")
        return False
    
    # Check TRL
    try:
        from trl import SFTTrainer, GRPOTrainer
        print("✅ TRL is installed")
    except ImportError:
        print("❌ TRL not installed")
        print("   Install with: pip install trl")
        return False
    
    # Check ALCHEMY modules
    try:
        from models import UnslothModelLoader, check_unsloth_available
        from agent import UnslothRLTrainer, UnslothRewardFunctions
        print("✅ ALCHEMY Unsloth modules are available")
    except ImportError as e:
        print(f"❌ ALCHEMY modules import error: {e}")
        return False
    
    return True


# =============================================================================
# EXAMPLE 1: Basic Model Loading with Unsloth
# =============================================================================

def example_model_loading():
    """
    Demonstrates loading a model with Unsloth optimizations.
    
    Key benefits:
    - 4x faster download with pre-quantized models
    - 70% less VRAM usage
    - Automatic LoRA configuration
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Model Loading with Unsloth")
    print("="*70 + "\n")
    
    from models import (
        UnslothModelLoader,
        UnslothConfig,
        UnslothPrecision,
        create_unsloth_loader,
        get_recommended_model,
    )
    
    # --- Get recommended model based on VRAM ---
    print("1. Getting recommended model for 8GB VRAM...")
    recommended = get_recommended_model(task="coding", vram_gb=8)
    print(f"   Recommended: {recommended}")
    
    # --- Show pre-quantized model aliases ---
    print("\n2. Available pre-quantized model aliases:")
    aliases = list(UnslothModelLoader.PREQANTIZED_MODELS.keys())[:5]
    for alias in aliases:
        full_name = UnslothModelLoader.get_prequantized_model(alias)
        print(f"   '{alias}' -> {full_name}")
    
    # --- Configuration example ---
    print("\n3. Creating Unsloth configuration...")
    config = UnslothConfig(
        model_name="unsloth/Mistral-7B-v0.3-bnb-4bit",
        precision=UnslothPrecision.FOUR_BIT,
        max_seq_length=2048,
        lora_r=16,
        lora_alpha=16,
        use_gradient_checkpointing="unsloth",  # 30% more VRAM savings!
    )
    
    print(f"   Model: {config.model_name}")
    print(f"   Precision: {config.precision.value}")
    print(f"   Max sequence length: {config.max_seq_length}")
    print(f"   LoRA r: {config.lora_r}")
    print(f"   Gradient checkpointing: {config.use_gradient_checkpointing}")
    
    # --- Factory function example ---
    print("\n4. Using factory function (recommended):")
    print("""
    loader = create_unsloth_loader(
        model_name="mistral-7b",      # Uses alias
        precision="4bit",
        max_seq_length=4096,
        lora_r=32,
    )
    model, tokenizer = loader.load()
    """)
    
    print("\n✅ Model loading example complete!")
    print("   (Model not actually loaded to save memory)")


# =============================================================================
# EXAMPLE 2: Supervised Fine-Tuning (SFT) with Unsloth
# =============================================================================

def example_sft_training():
    """
    Demonstrates SFT training with Unsloth.
    
    This is the basic fine-tuning approach before RL.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Supervised Fine-Tuning (SFT)")
    print("="*70 + "\n")
    
    print("SFT Training Code Example:")
    print("-" * 50)
    
    code = '''
from models import create_unsloth_loader
from agent import create_unsloth_rl_trainer
from datasets import load_dataset

# 1. Load model with Unsloth (2x faster, 70% less VRAM)
loader = create_unsloth_loader(
    model_name="mistral-7b",
    precision="4bit",
    max_seq_length=2048,
)
model, tokenizer = loader.load()

# 2. Load dataset
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

# 3. Create SFT trainer
trainer = create_unsloth_rl_trainer(
    model=model,
    tokenizer=tokenizer,
    algorithm="sft",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    optim="adamw_8bit",  # Memory efficient optimizer
    output_dir="./outputs/sft",
)

# 4. Train
trainer.train(dataset)

# 5. Save model
trainer.save("./outputs/sft", save_method="lora")
'''
    
    print(code)
    print("\n✅ SFT example complete!")


# =============================================================================
# EXAMPLE 3: GRPO (Reinforcement Learning) with Unsloth
# =============================================================================

def example_grpo_training():
    """
    Demonstrates GRPO training with Unsloth.
    
    GRPO (Group Relative Policy Optimization) is the same algorithm
    used by DeepSeek for training their reasoning models.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: GRPO (Reinforcement Learning)")
    print("="*70 + "\n")
    
    from agent import UnslothRewardFunctions
    
    # --- Show available reward functions ---
    print("1. Available Reward Functions:")
    print("   - correctness_reward: Based on keyword overlap")
    print("   - coding_reward: Checks syntax, docstrings, type hints")
    print("   - reasoning_reward: Checks for step-by-step patterns")
    print("   - create_combined_reward(): Mix all rewards")
    
    # --- Demonstrate reward function ---
    print("\n2. Testing Reward Functions:")
    
    # Test coding reward
    test_completions = [
        '```python\ndef hello(): print("Hi")\n```',
        '```python\ndef add(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b\n```',
        'This is not code at all',
    ]
    
    rewards = UnslothRewardFunctions.coding_reward(
        prompts=["Write code"] * 3,
        completions=test_completions,
    )
    
    print("   Coding rewards for test completions:")
    for i, (comp, reward) in enumerate(zip(test_completions, rewards)):
        preview = comp[:40].replace('\n', ' ') + "..."
        print(f"   {i+1}. '{preview}' -> reward: {reward:.2f}")
    
    # --- GRPO training code ---
    print("\n3. GRPO Training Code Example:")
    print("-" * 50)
    
    code = '''
from models import create_unsloth_loader
from agent import (
    create_unsloth_rl_trainer,
    UnslothRewardFunctions,
)
from datasets import load_dataset

# 1. Load model
loader = create_unsloth_loader("qwen-2.5-7b", precision="4bit")
model, tokenizer = loader.load()

# 2. Load reasoning dataset
dataset = load_dataset("gsm8k", "main", split="train")

# 3. Create custom reward function
reward_fn = UnslothRewardFunctions.create_combined_reward(
    coding_weight=0.2,
    reasoning_weight=0.5,
    correctness_weight=0.3,
)

# 4. Create GRPO trainer
trainer = create_unsloth_rl_trainer(
    model=model,
    tokenizer=tokenizer,
    algorithm="grpo",
    reward_fn=reward_fn,
    # GRPO-specific settings
    grpo_num_generations=4,
    grpo_temperature=0.7,
    grpo_max_new_tokens=512,
    # Training settings
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    output_dir="./outputs/grpo",
)

# 5. Train with GRPO
trainer.train(dataset)

# 6. Save in GGUF format for llama.cpp
trainer.save("./outputs/grpo", save_method="gguf")
'''
    
    print(code)
    print("\n✅ GRPO example complete!")


# =============================================================================
# EXAMPLE 4: DPO (Direct Preference Optimization)
# =============================================================================

def example_dpo_training():
    """
    Demonstrates DPO training with Unsloth.
    
    DPO is simpler than GRPO - it uses preference pairs
    (chosen vs rejected) instead of reward functions.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: DPO (Direct Preference Optimization)")
    print("="*70 + "\n")
    
    print("DPO Training Code Example:")
    print("-" * 50)
    
    code = '''
from models import create_unsloth_loader
from agent import create_unsloth_rl_trainer
from datasets import load_dataset

# 1. Load model
loader = create_unsloth_loader("llama-3.1-8b", precision="4bit")
model, tokenizer = loader.load()

# 2. Load preference dataset
# DPO requires: prompt, chosen, rejected columns
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

# 3. Create DPO trainer
trainer = create_unsloth_rl_trainer(
    model=model,
    tokenizer=tokenizer,
    algorithm="dpo",
    dpo_beta=0.1,  # DPO-specific parameter
    num_train_epochs=1,
    per_device_train_batch_size=2,
    output_dir="./outputs/dpo",
)

# 4. Train
trainer.train(dataset)

# 5. Save
trainer.save("./outputs/dpo", save_method="merged_16bit")
'''
    
    print(code)
    print("\n✅ DPO example complete!")


# =============================================================================
# EXAMPLE 5: Long Context Training
# =============================================================================

def example_long_context():
    """
    Demonstrates Unsloth's advantage for long context training.
    
    Unsloth enables 10-13x longer context lengths than standard training.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Long Context Training")
    print("="*70 + "\n")
    
    print("Context Length Comparison (Llama 3.1 8B with QLoRA):")
    print("-" * 50)
    
    context_table = """
    | GPU VRAM | Unsloth Context | HuggingFace + FA2 |
    |----------|-----------------|-------------------|
    | 8 GB     | 2,972           | OOM               |
    | 12 GB    | 21,848          | 932               |
    | 16 GB    | 40,724          | 2,551             |
    | 24 GB    | 78,475          | 5,789             |
    | 48 GB    | 191,728         | 15,502            |
    | 80 GB    | 342,733         | 28,454            |
    """
    print(context_table)
    
    print("\nLong Context Training Code:")
    print("-" * 50)
    
    code = '''
from models import create_unsloth_loader

# Enable very long context (e.g., for document processing)
loader = create_unsloth_loader(
    model_name="llama-3.1-8b",
    precision="4bit",
    max_seq_length=32768,  # 32K context!
    use_gradient_checkpointing="unsloth",  # Critical for long context
)
model, tokenizer = loader.load()

# Now you can train on very long documents
'''
    
    print(code)
    print("\n✅ Long context example complete!")


# =============================================================================
# EXAMPLE 6: Saving and Export Options
# =============================================================================

def example_saving():
    """
    Demonstrates various model saving options with Unsloth.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Saving and Export Options")
    print("="*70 + "\n")
    
    print("Available Save Methods:")
    print("-" * 50)
    
    methods = """
    1. "lora" (default)
       - Saves only LoRA adapters (~50MB)
       - Smallest file size
       - Requires base model to load
    
    2. "merged_16bit"
       - Merges LoRA into base model
       - Saves in 16-bit precision
       - Can be used directly with HuggingFace
    
    3. "merged_4bit"
       - Merges LoRA and saves in 4-bit
       - Smaller than 16-bit
       - Good for deployment
    
    4. "gguf"
       - Saves as GGUF format
       - Compatible with llama.cpp, Ollama
       - Multiple quantization options (q4_k_m, q5_k_m, q8_0)
    """
    print(methods)
    
    print("Code Examples:")
    print("-" * 50)
    
    code = '''
# Save LoRA adapter only
loader.save_model("./model_lora", save_method="lora")

# Merge and save in 16-bit
loader.save_model("./model_merged", save_method="merged_16bit")

# Save as GGUF for llama.cpp/Ollama
loader.save_model(
    "./model_gguf",
    save_method="gguf",
    quantization_method="q4_k_m",  # or q5_k_m, q8_0
)

# Push to HuggingFace Hub
loader.save_model(
    "./model",
    save_method="merged_16bit",
    push_to_hub=True,
    hub_model_id="username/my-fine-tuned-model",
)
'''
    
    print(code)
    print("\n✅ Saving example complete!")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "#"*70)
    print("#" + " "*20 + "UNSLOTH INTEGRATION DEMO" + " "*22 + "#")
    print("#"*70)
    
    print("""
This demo showcases Unsloth integration in ALCHEMY:

🦥 Unsloth provides:
   • 2x faster training speed
   • 70% less VRAM usage
   • 10-13x longer context lengths
   • Native RL support (GRPO, DPO, ORPO, etc.)

📚 Reference: https://github.com/unslothai/unsloth
""")
    
    # Check dependencies (but don't require them for the demo)
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n⚠️  Some dependencies are missing.")
        print("   The demo will show code examples but won't run actual training.\n")
    
    # Run examples
    example_model_loading()
    example_sft_training()
    example_grpo_training()
    example_dpo_training()
    example_long_context()
    example_saving()
    
    print("\n" + "#"*70)
    print("#" + " "*22 + "ALL EXAMPLES COMPLETE!" + " "*23 + "#")
    print("#"*70)
    
    print("""
Next Steps:
1. Install Unsloth: pip install unsloth
2. Choose a model: get_recommended_model(task="coding", vram_gb=8)
3. Load with: create_unsloth_loader("mistral-7b")
4. Train with GRPO: create_unsloth_rl_trainer(algorithm="grpo")
5. Save as GGUF for deployment
""")

