"""
Meta-Agent: Agents that generate other agents.

Inspired by PocketFlow's minimalist philosophy (~100 lines core),
this module implements a meta-agent system that can dynamically:
- Generate specialized agent configurations
- Create task-specific reward functions
- Produce SOPs (Standard Operating Procedures) on-demand
- Spawn child agents for specific subtasks

The key insight from PocketFlow is that agents can be self-replicating
and self-specializing, reducing the need for manual configuration.

References:
- PocketFlow: https://github.com/The-Pocket/PocketFlow
- Pattern: "Agents building agents" for autonomous scaling
"""

from typing import Optional, Dict, Any, List, Callable, Union, TypeVar
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import logging
import re
import copy

logger = logging.getLogger(__name__)

# Type variable for generic agent configurations
T = TypeVar("T")


class AgentType(str, Enum):
    """Types of agents that can be generated."""
    CODING = "coding"
    REASONING = "reasoning"
    FUNCTION_CALLING = "function_calling"
    CHAT = "chat"
    RAG = "rag"
    CUSTOM = "custom"


@dataclass
class AgentBlueprint:
    """
    Blueprint for generating a specialized agent.
    
    This is the "DNA" that defines an agent's behavior,
    capabilities, and configuration.
    """
    name: str
    agent_type: AgentType
    description: str
    
    # Model configuration
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 512
    
    # LoRA configuration for specialization
    lora_r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    
    # Reward function configuration
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "correctness": 0.5,
        "quality": 0.3,
        "efficiency": 0.2,
    })
    
    # System prompt template
    system_prompt: str = "You are a helpful AI assistant."
    
    # Tools this agent can use
    available_tools: List[str] = field(default_factory=list)
    
    # SOP triggers and procedures
    sop_triggers: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert blueprint to dictionary."""
        return {
            "name": self.name,
            "agent_type": self.agent_type.value,
            "description": self.description,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "reward_weights": self.reward_weights,
            "system_prompt": self.system_prompt,
            "available_tools": self.available_tools,
            "sop_triggers": self.sop_triggers,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentBlueprint":
        """Create blueprint from dictionary."""
        agent_type = data.get("agent_type", "custom")
        if isinstance(agent_type, str):
            agent_type = AgentType(agent_type)
        
        return cls(
            name=data["name"],
            agent_type=agent_type,
            description=data["description"],
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.95),
            max_new_tokens=data.get("max_new_tokens", 512),
            lora_r=data.get("lora_r", 16),
            lora_alpha=data.get("lora_alpha", 32),
            target_modules=data.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            reward_weights=data.get("reward_weights", {}),
            system_prompt=data.get("system_prompt", "You are a helpful AI assistant."),
            available_tools=data.get("available_tools", []),
            sop_triggers=data.get("sop_triggers", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GeneratedSOP:
    """A dynamically generated Standard Operating Procedure."""
    name: str
    description: str
    trigger: str
    steps: List[Dict[str, Any]]
    category: str = "generated"
    priority: int = 5
    
    def to_sop_dict(self) -> Dict[str, Any]:
        """Convert to SOP-compatible dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "trigger": self.trigger,
            "category": self.category,
            "priority": self.priority,
            "steps": self.steps,
            "metadata": {"generated": True},
        }


# =============================================================================
# META-AGENT: The agent that generates agents
# =============================================================================

class MetaAgent:
    """
    Meta-Agent: An agent that generates other agents.
    
    Inspired by PocketFlow's minimalist "agents building agents" pattern.
    
    Capabilities:
    - Analyze task descriptions to determine optimal agent configuration
    - Generate specialized AgentBlueprints for specific tasks
    - Create custom reward functions based on task requirements
    - Produce SOPs dynamically for new procedures
    - Spawn and manage child agents
    
    Example:
        ```python
        meta = MetaAgent(base_model, tokenizer)
        
        # Generate a specialized coding agent
        blueprint = meta.generate_agent_blueprint(
            task="Write and debug Python functions with type hints"
        )
        
        # Generate a custom SOP
        sop = meta.generate_sop(
            task="Code review process for security-critical code"
        )
        ```
    """
    
    # Templates for agent generation
    AGENT_TEMPLATES: Dict[AgentType, AgentBlueprint] = {
        AgentType.CODING: AgentBlueprint(
            name="coding_agent",
            agent_type=AgentType.CODING,
            description="Specialized agent for writing and debugging code",
            temperature=0.3,
            max_new_tokens=1024,
            reward_weights={"syntax": 0.3, "functionality": 0.4, "style": 0.2, "docs": 0.1},
            system_prompt="""You are an expert Python developer. You write clean, 
efficient, and well-documented code. Always include type hints and docstrings.""",
            sop_triggers=["write code", "implement", "debug", "fix bug"],
        ),
        AgentType.REASONING: AgentBlueprint(
            name="reasoning_agent",
            agent_type=AgentType.REASONING,
            description="Agent specialized in step-by-step reasoning",
            temperature=0.5,
            max_new_tokens=2048,
            reward_weights={"correctness": 0.5, "reasoning_steps": 0.3, "clarity": 0.2},
            system_prompt="""You are a logical reasoning expert. Break down problems 
into steps, show your work, and verify your conclusions.""",
            sop_triggers=["solve", "reason", "analyze", "calculate"],
        ),
        AgentType.FUNCTION_CALLING: AgentBlueprint(
            name="tool_agent",
            agent_type=AgentType.FUNCTION_CALLING,
            description="Agent for invoking tools and APIs",
            temperature=0.2,
            max_new_tokens=512,
            reward_weights={"valid_json": 0.4, "correct_tool": 0.3, "correct_args": 0.3},
            system_prompt="""You are a tool-use specialist. Determine which tool to use 
and provide correctly formatted function calls.""",
            available_tools=["search_knowledge_base", "execute_python", "web_search"],
            sop_triggers=["use tool", "call function", "search", "execute"],
        ),
        AgentType.RAG: AgentBlueprint(
            name="rag_agent",
            agent_type=AgentType.RAG,
            description="Agent for retrieval-augmented generation",
            temperature=0.4,
            max_new_tokens=1024,
            reward_weights={"relevance": 0.4, "accuracy": 0.4, "citation": 0.2},
            system_prompt="""You are a knowledge assistant. Search for relevant information 
and provide accurate answers with citations.""",
            available_tools=["search_knowledge_base"],
            sop_triggers=["find", "search", "what is", "documentation"],
        ),
        AgentType.CHAT: AgentBlueprint(
            name="chat_agent",
            agent_type=AgentType.CHAT,
            description="General-purpose conversational agent",
            temperature=0.7,
            max_new_tokens=512,
            reward_weights={"helpfulness": 0.4, "coherence": 0.3, "engagement": 0.3},
            system_prompt="You are a helpful, friendly AI assistant.",
            sop_triggers=["explain", "help", "what", "how"],
        ),
    }
    
    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        use_llm_generation: bool = False,
    ):
        """
        Initialize the Meta-Agent.
        
        Args:
            model: Optional LLM for advanced generation
            tokenizer: Tokenizer for the model
            use_llm_generation: Whether to use LLM for dynamic generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.use_llm_generation = use_llm_generation and model is not None
        
        # Registry of generated agents
        self.generated_agents: Dict[str, AgentBlueprint] = {}
        
        # Registry of generated SOPs
        self.generated_sops: Dict[str, GeneratedSOP] = {}
        
        logger.info("MetaAgent initialized")
    
    def analyze_task(self, task_description: str) -> Dict[str, Any]:
        """
        Analyze a task description to extract requirements.
        
        Args:
            task_description: Natural language description of the task
            
        Returns:
            Dictionary with extracted task characteristics
        """
        task_lower = task_description.lower()
        
        analysis = {
            "detected_type": AgentType.CHAT,
            "complexity": "medium",
            "requires_tools": False,
            "requires_reasoning": False,
            "requires_search": False,
            "keywords": [],
        }
        
        # Detect agent type based on keywords
        coding_keywords = ["code", "python", "function", "implement", "debug", "program", "script"]
        reasoning_keywords = ["solve", "calculate", "reason", "prove", "analyze", "math"]
        tool_keywords = ["call", "api", "execute", "tool", "function call"]
        rag_keywords = ["search", "find", "documentation", "knowledge", "retrieve"]
        
        for kw in coding_keywords:
            if kw in task_lower:
                analysis["detected_type"] = AgentType.CODING
                analysis["keywords"].append(kw)
        
        for kw in reasoning_keywords:
            if kw in task_lower:
                analysis["detected_type"] = AgentType.REASONING
                analysis["requires_reasoning"] = True
                analysis["keywords"].append(kw)
        
        for kw in tool_keywords:
            if kw in task_lower:
                analysis["detected_type"] = AgentType.FUNCTION_CALLING
                analysis["requires_tools"] = True
                analysis["keywords"].append(kw)
        
        for kw in rag_keywords:
            if kw in task_lower:
                analysis["requires_search"] = True
                analysis["keywords"].append(kw)
                if analysis["detected_type"] == AgentType.CHAT:
                    analysis["detected_type"] = AgentType.RAG
        
        # Estimate complexity
        word_count = len(task_description.split())
        if word_count > 50 or "complex" in task_lower or "advanced" in task_lower:
            analysis["complexity"] = "high"
        elif word_count < 15 or "simple" in task_lower or "basic" in task_lower:
            analysis["complexity"] = "low"
        
        return analysis
    
    def generate_agent_blueprint(
        self,
        task: str,
        name: Optional[str] = None,
        customize: Optional[Dict[str, Any]] = None,
    ) -> AgentBlueprint:
        """
        Generate a specialized agent blueprint for a task.
        
        This is the core "agent generating agent" functionality.
        
        Args:
            task: Description of the task the agent should handle
            name: Optional custom name for the agent
            customize: Optional customizations to apply
            
        Returns:
            AgentBlueprint configured for the task
        """
        # Analyze the task
        analysis = self.analyze_task(task)
        detected_type = analysis["detected_type"]
        
        # Start from template
        template = self.AGENT_TEMPLATES.get(detected_type, self.AGENT_TEMPLATES[AgentType.CHAT])
        blueprint = copy.deepcopy(template)
        
        # Customize name
        if name:
            blueprint.name = name
        else:
            blueprint.name = f"{detected_type.value}_agent_{len(self.generated_agents)}"
        
        # Customize description
        blueprint.description = f"Agent specialized for: {task}"
        
        # Adjust parameters based on complexity
        if analysis["complexity"] == "high":
            blueprint.max_new_tokens = min(blueprint.max_new_tokens * 2, 4096)
            blueprint.lora_r = 32
            blueprint.lora_alpha = 64
        elif analysis["complexity"] == "low":
            blueprint.max_new_tokens = max(blueprint.max_new_tokens // 2, 256)
            blueprint.temperature = min(blueprint.temperature + 0.1, 1.0)
        
        # Add tools if needed
        if analysis["requires_search"] and "search_knowledge_base" not in blueprint.available_tools:
            blueprint.available_tools.append("search_knowledge_base")
        
        if analysis["requires_tools"] and "execute_python" not in blueprint.available_tools:
            blueprint.available_tools.append("execute_python")
        
        # Add task-specific triggers to SOP
        blueprint.sop_triggers.extend(analysis["keywords"])
        
        # Apply custom overrides
        if customize:
            for key, value in customize.items():
                if hasattr(blueprint, key):
                    setattr(blueprint, key, value)
        
        # Update system prompt with task context
        blueprint.system_prompt += f"\n\nYour primary task: {task}"
        
        # Store in registry
        self.generated_agents[blueprint.name] = blueprint
        
        logger.info(f"Generated agent blueprint: {blueprint.name} (type: {detected_type.value})")
        
        return blueprint
    
    def generate_sop(
        self,
        task: str,
        name: Optional[str] = None,
        num_steps: int = 5,
    ) -> GeneratedSOP:
        """
        Generate a Standard Operating Procedure for a task.
        
        Args:
            task: Description of the procedure needed
            name: Optional name for the SOP
            num_steps: Approximate number of steps
            
        Returns:
            GeneratedSOP ready to be used
        """
        # Analyze task to determine category
        analysis = self.analyze_task(task)
        
        # Generate SOP name
        if not name:
            task_words = re.findall(r'\b\w+\b', task.lower())[:3]
            name = "_".join(task_words) + "_procedure"
        
        # Determine category based on detected type
        category_map = {
            AgentType.CODING: "coding",
            AgentType.REASONING: "reasoning",
            AgentType.FUNCTION_CALLING: "tools",
            AgentType.RAG: "rag",
            AgentType.CHAT: "general",
        }
        category = category_map.get(analysis["detected_type"], "general")
        
        # Generate steps based on task type
        steps = self._generate_steps_for_task(task, analysis, num_steps)
        
        # Create trigger from keywords
        trigger = ", ".join(analysis["keywords"][:5]) if analysis["keywords"] else task[:50]
        
        sop = GeneratedSOP(
            name=name,
            description=f"Procedure for: {task}",
            trigger=trigger,
            steps=steps,
            category=category,
            priority=7 if analysis["complexity"] == "high" else 5,
        )
        
        # Store in registry
        self.generated_sops[name] = sop
        
        logger.info(f"Generated SOP: {name} with {len(steps)} steps")
        
        return sop
    
    def _generate_steps_for_task(
        self,
        task: str,
        analysis: Dict[str, Any],
        num_steps: int,
    ) -> List[Dict[str, Any]]:
        """Generate procedure steps based on task analysis."""
        steps = []
        detected_type = analysis["detected_type"]
        
        # Base steps that apply to most procedures
        steps.append({
            "action": "Understand and clarify the requirements",
            "expected_output": "Clear understanding of what needs to be done",
        })
        
        if detected_type == AgentType.CODING:
            steps.extend([
                {"action": "Design the solution structure and interfaces"},
                {"action": "Implement the core functionality with proper error handling"},
                {"action": "Add type hints and documentation"},
                {"action": "Write unit tests for the implementation"},
                {"action": "Review and refactor if needed"},
            ])
        elif detected_type == AgentType.REASONING:
            steps.extend([
                {"action": "Break down the problem into smaller parts"},
                {"action": "Identify relevant information and constraints"},
                {"action": "Apply logical reasoning step-by-step"},
                {"action": "Verify each step of the reasoning"},
                {"action": "Synthesize the final answer with explanation"},
            ])
        elif detected_type == AgentType.FUNCTION_CALLING:
            steps.extend([
                {"action": "Identify which tool or function is needed"},
                {"action": "Extract required parameters from the context"},
                {"action": "Format the function call with correct arguments"},
                {"action": "Execute the function call"},
                {"action": "Process and present the results"},
            ])
        elif detected_type == AgentType.RAG:
            steps.extend([
                {"action": "Formulate an effective search query"},
                {"action": "Search the knowledge base using search_knowledge_base", "tools": ["search_knowledge_base"]},
                {"action": "Evaluate relevance of retrieved documents"},
                {"action": "Synthesize answer from relevant sources"},
                {"action": "Cite sources and acknowledge gaps if any"},
            ])
        else:  # General/Chat
            steps.extend([
                {"action": "Analyze the user's request thoroughly"},
                {"action": "Gather relevant information"},
                {"action": "Formulate a helpful response"},
                {"action": "Ensure the response is clear and complete"},
            ])
        
        # Truncate or pad to requested number
        steps = steps[:num_steps]
        
        return steps
    
    def generate_reward_function(
        self,
        task: str,
        blueprint: Optional[AgentBlueprint] = None,
    ) -> Callable[[str, str, Optional[str]], float]:
        """
        Generate a custom reward function for a task.
        
        Args:
            task: Task description
            blueprint: Optional blueprint to use for weights
            
        Returns:
            Callable reward function
        """
        if blueprint is None:
            blueprint = self.generate_agent_blueprint(task)
        
        weights = blueprint.reward_weights
        agent_type = blueprint.agent_type
        
        def custom_reward(
            prompt: str,
            generation: str,
            reference: Optional[str] = None,
        ) -> float:
            """Dynamically generated reward function."""
            reward = 0.0
            
            # Length check (universal)
            gen_len = len(generation.strip())
            if gen_len < 10:
                return -0.5
            elif gen_len > 50:
                reward += 0.1
            
            # Type-specific rewards
            if agent_type == AgentType.CODING:
                # Syntax check
                code_blocks = re.findall(r'```(?:python)?\n?(.*?)```', generation, re.DOTALL)
                if code_blocks:
                    try:
                        compile(code_blocks[0], '<string>', 'exec')
                        reward += weights.get("syntax", 0.3)
                    except SyntaxError:
                        reward -= 0.3
                
                # Documentation
                if '"""' in generation or "'''" in generation:
                    reward += weights.get("docs", 0.1)
                
                # Type hints
                if '->' in generation and ':' in generation:
                    reward += weights.get("style", 0.1)
            
            elif agent_type == AgentType.REASONING:
                # Step markers
                step_patterns = [r'step \d', r'\d\.', r'first', r'then', r'finally', r'therefore']
                reasoning_score = sum(1 for p in step_patterns if re.search(p, generation.lower()))
                reward += min(weights.get("reasoning_steps", 0.3), reasoning_score * 0.05)
            
            elif agent_type == AgentType.FUNCTION_CALLING:
                # Valid JSON
                try:
                    fc_match = re.search(r'\{[^{}]*\}', generation)
                    if fc_match:
                        json.loads(fc_match.group())
                        reward += weights.get("valid_json", 0.3)
                except json.JSONDecodeError:
                    reward -= 0.2
            
            # Reference comparison (if available)
            if reference:
                ref_words = set(re.findall(r'\b\w+\b', reference.lower()))
                gen_words = set(re.findall(r'\b\w+\b', generation.lower()))
                if ref_words:
                    overlap = len(ref_words & gen_words) / len(ref_words)
                    reward += weights.get("correctness", 0.5) * overlap
            
            return max(-1.0, min(1.0, reward))
        
        return custom_reward
    
    def spawn_child_agent(
        self,
        parent_blueprint: AgentBlueprint,
        specialization: str,
    ) -> AgentBlueprint:
        """
        Spawn a child agent from a parent with additional specialization.
        
        This enables hierarchical agent generation.
        
        Args:
            parent_blueprint: The parent agent's blueprint
            specialization: Additional specialization for the child
            
        Returns:
            New AgentBlueprint for the child agent
        """
        child = copy.deepcopy(parent_blueprint)
        child.name = f"{parent_blueprint.name}_child_{len(self.generated_agents)}"
        child.description = f"{parent_blueprint.description} + {specialization}"
        child.system_prompt += f"\n\nAdditional specialization: {specialization}"
        child.metadata["parent"] = parent_blueprint.name
        child.metadata["specialization"] = specialization
        
        # Analyze specialization for further customization
        analysis = self.analyze_task(specialization)
        child.sop_triggers.extend(analysis["keywords"])
        
        self.generated_agents[child.name] = child
        
        logger.info(f"Spawned child agent: {child.name} from {parent_blueprint.name}")
        
        return child
    
    def list_generated_agents(self) -> List[Dict[str, Any]]:
        """List all generated agent blueprints."""
        return [
            {"name": bp.name, "type": bp.agent_type.value, "description": bp.description}
            for bp in self.generated_agents.values()
        ]
    
    def list_generated_sops(self) -> List[Dict[str, Any]]:
        """List all generated SOPs."""
        return [
            {"name": sop.name, "category": sop.category, "description": sop.description}
            for sop in self.generated_sops.values()
        ]
    
    def export_blueprint(self, name: str, filepath: str) -> None:
        """Export an agent blueprint to JSON."""
        if name not in self.generated_agents:
            raise ValueError(f"Agent '{name}' not found")
        
        blueprint = self.generated_agents[name]
        with open(filepath, 'w') as f:
            json.dump(blueprint.to_dict(), f, indent=2)
        
        logger.info(f"Exported blueprint to: {filepath}")
    
    def export_sop(self, name: str, filepath: str) -> None:
        """Export a generated SOP to JSON."""
        if name not in self.generated_sops:
            raise ValueError(f"SOP '{name}' not found")
        
        sop = self.generated_sops[name]
        with open(filepath, 'w') as f:
            json.dump(sop.to_sop_dict(), f, indent=2)
        
        logger.info(f"Exported SOP to: {filepath}")


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_meta_agent(
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    use_llm: bool = False,
) -> MetaAgent:
    """
    Factory function to create a MetaAgent.
    
    Args:
        model: Optional LLM model
        tokenizer: Optional tokenizer
        use_llm: Whether to use LLM for advanced generation
        
    Returns:
        Configured MetaAgent
    """
    return MetaAgent(
        model=model,
        tokenizer=tokenizer,
        use_llm_generation=use_llm,
    )


def quick_generate_agent(task: str) -> AgentBlueprint:
    """
    Quick helper to generate an agent blueprint without instantiating MetaAgent.
    
    Args:
        task: Task description
        
    Returns:
        AgentBlueprint for the task
    """
    meta = MetaAgent()
    return meta.generate_agent_blueprint(task)

