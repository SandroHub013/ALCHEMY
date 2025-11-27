"""Moduli per gli agenti di training."""

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

# Tools per Function Calling
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
]
