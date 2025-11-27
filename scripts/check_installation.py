#!/usr/bin/env python3
"""
Script di verifica installazione.

Esegui dopo l'installazione per verificare che tutto funzioni:
    python scripts/check_installation.py
"""

import sys
from typing import Tuple

def check_import(module: str, package: str = None) -> Tuple[bool, str]:
    """Verifica se un modulo è importabile."""
    try:
        __import__(module)
        return True, "✅"
    except ImportError as e:
        return False, f"❌ ({e})"

def main():
    print("=" * 60)
    print("🔍 VERIFICA INSTALLAZIONE LLM Fine-tuning Agent Lightning")
    print("=" * 60)
    print()
    
    # Verifica Python
    py_version = sys.version_info
    print(f"Python: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version < (3, 10):
        print("⚠️  Richiesto Python >= 3.10")
    else:
        print("✅ Versione Python OK")
    print()
    
    # Lista moduli da verificare
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
    
    print("📦 DIPENDENZE CORE:")
    print("-" * 40)
    
    all_ok = True
    for module, name in modules:
        ok, status = check_import(module)
        print(f"  {name:30} {status}")
        if not ok:
            all_ok = False
    
    print()
    
    # Verifica Agent Lightning (OBBLIGATORIO)
    print("📦 AGENT LIGHTNING (core):")
    print("-" * 40)
    ok, status = check_import("agentlightning")
    print(f"  {'Agent Lightning':30} {status}")
    if not ok:
        print("  ❌ Agent Lightning è OBBLIGATORIO!")
        print("     Installa con: pip install agentlightning")
        all_ok = False
    print()
    
    # Verifica CUDA
    print("🎮 GPU/CUDA:")
    print("-" * 40)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA disponibile")
            print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  ✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  ⚠️  CUDA non disponibile (training su CPU sarà lento)")
    except Exception as e:
        print(f"  ❌ Errore verifica CUDA: {e}")
    print()
    
    # Verifica moduli locali
    print("📂 MODULI PROGETTO:")
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
        print("✅ INSTALLAZIONE COMPLETATA CON SUCCESSO!")
        print()
        print("Prossimi passi:")
        print("  1. python main.py --config config/config.yaml")
        print("  2. python main_agent_lightning.py --config config/config.yaml")
    else:
        print("⚠️  ALCUNE DIPENDENZE MANCANO")
        print()
        print("Prova:")
        print("  pip install -e .")
        print()
        print("Per problemi con bitsandbytes su Windows:")
        print("  pip install bitsandbytes-windows")
    
    print("=" * 60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())

