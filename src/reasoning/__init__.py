"""
Modulo Reasoning - Integrazione LUFFY e Search-R1.

Questo modulo fornisce capacità avanzate di ragionamento:
- LUFFY: Off-policy learning per migliorare le capacità di reasoning
- Search-R1: Ragionamento con integrazione di ricerca web/knowledge base
- ExGRPO: Apprendimento dall'esperienza del modello stesso

Riferimenti:
- LUFFY Paper: https://arxiv.org/abs/2504.14945
- Search-R1: https://github.com/PeterGriffinJin/Search-R1
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
"""

from .luffy_trainer import (
    LuffyTrainer,
    LuffyConfig,
    ExGRPOConfig,
    OffPolicyDataMixer,
)

from .search_r1 import (
    SearchR1Trainer,
    SearchR1Config,
    SearchEngine,
    ReasoningWithSearch,
)

__all__ = [
    # LUFFY
    "LuffyTrainer",
    "LuffyConfig", 
    "ExGRPOConfig",
    "OffPolicyDataMixer",
    # Search-R1
    "SearchR1Trainer",
    "SearchR1Config",
    "SearchEngine",
    "ReasoningWithSearch",
]

