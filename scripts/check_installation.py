#!/usr/bin/env python3
"""
Installation verification script.

Run after installation to verify everything works:
    python scripts/check_installation.py
"""

import sys
from typing import Tuple

def check_import(module: str, package: str = None) -> Tuple[bool, str]:
    """Check if a module is importable."""
    try:
        __import__(module)
        return True, "✅"
    except ImportError as e:
        return False, f"❌ ({e})"

def main():
    print("=" * 60)
    print("🔍 LLM Fine-tuning Agent Lightning INSTALLATION CHECK")
    print("=" * 60)
    print()
    
    # Check Python
    py_version = sys.version_info
    print(f"Python: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version < (3, 10):
        print("⚠️  Python >= 3.10 required")
    else:
        print("✅ Python version OK")
    print()
    
    # List of modules to check
    modules = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("accelerate", "Accelerate"),
        ("lightning", "PyTorch Lightning"),
        ("peft", "PEFT (LoRA)"),
        ("bitsandbytes", "bitsandbytes (4-bit)"),
        ("datasets", "HuggingFace Datasets"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "Sentence Transformers"),
        ("tree_sitter", "tree-sitter"),
        ("tree_sitter_python", "tree-sitter-python"),
        ("langchain", "LangChain"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
    ]
    
    print("📦 CORE DEPENDENCIES:")
    print("-" * 40)
    
    all_ok = True
    for module, name in modules:
        ok, status = check_import(module)
        print(f"  {name:30} {status}")
        if not ok:
            all_ok = False
    
    print()
    
    # Check Agent Lightning (REQUIRED)
    print("📦 AGENT LIGHTNING (core):")
    print("-" * 40)
    ok, status = check_import("agentlightning")
    print(f"  {'Agent Lightning':30} {status}")
    if not ok:
        print("  ❌ Agent Lightning is REQUIRED!")
        print("     Install with: pip install agentlightning")
        all_ok = False
    print()
    
    # Check CUDA
    print("🎮 GPU/CUDA:")
    print("-" * 40)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available")
            print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  ✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  ⚠️  CUDA not available (training on CPU will be slow)")
    except Exception as e:
        print(f"  ❌ CUDA check error: {e}")
    print()
    
    # Check local modules
    print("📂 PROJECT MODULES:")
    print("-" * 40)
    
    local_modules = [
        ("src.models", "ModelLoader"),
        ("src.agent", "Training Agent"),
        ("src.memory", "Memory System"),
        ("src.data", "Data Module"),
    ]
    
    for module, name in local_modules:
        ok, status = check_import(module)
        print(f"  {name:30} {status}")
        if not ok:
            all_ok = False
    
    print()
    print("=" * 60)
    
    if all_ok:
        print("✅ INSTALLATION COMPLETED SUCCESSFULLY!")
        print()
        print("Next steps:")
        print("  1. python main.py --config config/config.yaml")
        print("  2. python main_agent_lightning.py --config config/config.yaml")
    else:
        print("⚠️  SOME DEPENDENCIES ARE MISSING")
        print()
        print("Try:")
        print("  pip install -e .")
        print()
        print("For bitsandbytes issues on Windows:")
        print("  pip install bitsandbytes-windows")
    
    print("=" * 60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
