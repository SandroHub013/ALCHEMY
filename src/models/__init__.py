"""Modules for loading and managing models."""

from .model_loader import ModelLoader

# =============================================================================
# UNSLOTH - High-Performance Model Loading (2x faster, 70% less VRAM)
# =============================================================================
# Unsloth provides optimized model loading and training.
# Reference: https://github.com/unslothai/unsloth
from .unsloth_loader import (
    UnslothModelLoader,
    UnslothConfig,
    UnslothPrecision,
    create_unsloth_loader,
    load_with_unsloth,
    check_unsloth_available,
    get_recommended_model,
)

__all__ = [
    # Standard loader
    "ModelLoader",
    # Unsloth loader (high-performance)
    "UnslothModelLoader",
    "UnslothConfig",
    "UnslothPrecision",
    "create_unsloth_loader",
    "load_with_unsloth",
    "check_unsloth_available",
    "get_recommended_model",
]

