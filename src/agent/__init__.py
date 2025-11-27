"""Agent modules for training, orchestration, and meta-generation."""

from .training_agent import LLMTrainingAgent

# Agent Lightning integration
from .agent_lightning_trainer import (
    AgentLightningTrainer,
    AgentLightningConfig,
    TrainingAlgorithm,
    RewardFunction,
    create_agent_lightning_trainer,
    check_agent_lightning_available,
)

# Tools for Function Calling
from .tools import (
    Tool,
    ToolParameter,
    SEARCH_KNOWLEDGE_BASE,
    EXECUTE_PYTHON,
    WEB_SEARCH,
    ALL_TOOLS,
    DEFAULT_RAG_TOOLS,
    DEFAULT_AGENT_TOOLS,
    get_tools_json_schema,
    get_tools_prompt,
    format_tool_call,
    parse_tool_call,
    get_system_prompt_with_tools,
    SYSTEM_PROMPT_RAG,
)

# =============================================================================
# NEW: Meta-Agent (Inspired by PocketFlow)
# =============================================================================
# Agents that generate other agents, SOPs, and reward functions dynamically
from .meta_agent import (
    MetaAgent,
    AgentBlueprint,
    AgentType,
    GeneratedSOP,
    create_meta_agent,
    quick_generate_agent,
)

# =============================================================================
# NEW: Adaptive Trainer (Inspired by AgentFlow)
# =============================================================================
# Dynamic optimization during training with curriculum learning
from .adaptive_trainer import (
    AdaptiveTrainer,
    AdaptiveConfig,
    AdaptiveTrainingCallback,
    TrainingState,
    AdaptiveAction,
    CurriculumManager,
    MetricAnalyzer,
    create_adaptive_trainer,
    create_adaptive_callback,
)

# =============================================================================
# NEW: Swarm Trainer (Inspired by claude-flow)
# =============================================================================
# Multi-agent orchestration for distributed exploration
from .swarm_trainer import (
    SwarmTrainer,
    SwarmCoordinator,
    SwarmAgent,
    SwarmConfig,
    SwarmRole,
    Trajectory,
    create_swarm_trainer,
    create_swarm_coordinator,
)

# =============================================================================
# NEW: Unsloth RL Trainer (2x faster, 70% less VRAM)
# =============================================================================
# High-performance RL training using Unsloth optimizations
# Reference: https://github.com/unslothai/unsloth
from .unsloth_trainer import (
    UnslothRLTrainer,
    UnslothTrainerConfig,
    UnslothRLAlgorithm,
    UnslothRewardFunctions,
    create_unsloth_rl_trainer,
    quick_train_with_unsloth,
)

__all__ = [
    # PyTorch Lightning
    "LLMTrainingAgent",
    # Agent Lightning
    "AgentLightningTrainer",
    "AgentLightningConfig",
    "TrainingAlgorithm",
    "RewardFunction",
    "create_agent_lightning_trainer",
    "check_agent_lightning_available",
    # Tools
    "Tool",
    "ToolParameter",
    "SEARCH_KNOWLEDGE_BASE",
    "EXECUTE_PYTHON",
    "WEB_SEARCH",
    "ALL_TOOLS",
    "DEFAULT_RAG_TOOLS",
    "DEFAULT_AGENT_TOOLS",
    "get_tools_json_schema",
    "get_tools_prompt",
    "format_tool_call",
    "parse_tool_call",
    "get_system_prompt_with_tools",
    "SYSTEM_PROMPT_RAG",
    # Meta-Agent (PocketFlow-inspired)
    "MetaAgent",
    "AgentBlueprint",
    "AgentType",
    "GeneratedSOP",
    "create_meta_agent",
    "quick_generate_agent",
    # Adaptive Trainer (AgentFlow-inspired)
    "AdaptiveTrainer",
    "AdaptiveConfig",
    "AdaptiveTrainingCallback",
    "TrainingState",
    "AdaptiveAction",
    "CurriculumManager",
    "MetricAnalyzer",
    "create_adaptive_trainer",
    "create_adaptive_callback",
    # Swarm Trainer (claude-flow-inspired)
    "SwarmTrainer",
    "SwarmCoordinator",
    "SwarmAgent",
    "SwarmConfig",
    "SwarmRole",
    "Trajectory",
    "create_swarm_trainer",
    "create_swarm_coordinator",
    # Unsloth RL Trainer (high-performance)
    "UnslothRLTrainer",
    "UnslothTrainerConfig",
    "UnslothRLAlgorithm",
    "UnslothRewardFunctions",
    "create_unsloth_rl_trainer",
    "quick_train_with_unsloth",
]
