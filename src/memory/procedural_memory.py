"""
Procedural Memory through SOP (Standard Operating Procedures).

This module implements a procedural memory system that allows
the model to follow structured step-by-step procedures.

SOPs are composed of:
- Name and description
- Trigger (when to activate the procedure)
- Sequential steps with conditions
- Actions to execute
- Result validation

Usage:
    ```python
    from src.memory.procedural_memory import SOPManager, SOP, SOPStep
    
    # Create a SOP
    sop = SOP(
        name="debug_code",
        description="Procedure for debugging Python code",
        trigger="user asks to debug or find bugs",
        steps=[
            SOPStep(action="Read the code and identify the problem"),
            SOPStep(action="Propose a solution", condition="problem identified"),
            SOPStep(action="Verify the solution"),
        ]
    )
    
    # Use the manager
    manager = SOPManager()
    manager.add_sop(sop)
    
    # Find relevant SOP
    relevant = manager.find_relevant_sop("how do I debug this code?")
    ```
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    """Execution status of a step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class SOPStep:
    """
    Single step of a SOP.
    
    Attributes:
        action: Description of the action to execute
        condition: Condition to execute the step (None = always)
        expected_output: Expected output (for validation)
        tools: Tools to use in this step
        fallback: Alternative action if the step fails
    """
    action: str
    condition: Optional[str] = None
    expected_output: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    fallback: Optional[str] = None
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the step to dictionary."""
        return {
            "action": self.action,
            "condition": self.condition,
            "expected_output": self.expected_output,
            "tools": self.tools,
            "fallback": self.fallback,
            "status": self.status.value,
            "result": self.result,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SOPStep":
        """Create a step from dictionary."""
        status = data.get("status", "pending")
        if isinstance(status, str):
            status = StepStatus(status)
        
        return cls(
            action=data["action"],
            condition=data.get("condition"),
            expected_output=data.get("expected_output"),
            tools=data.get("tools", []),
            fallback=data.get("fallback"),
            status=status,
            result=data.get("result"),
        )


@dataclass
class SOP:
    """
    Standard Operating Procedure (SOP).
    
    A structured procedure that guides the model through
    a series of steps to complete a complex task.
    
    Attributes:
        name: Unique name of the SOP
        description: Description of the procedure
        trigger: Pattern/keywords that activate this SOP
        category: Category (coding, debugging, documentation, etc.)
        steps: List of steps to execute
        priority: Priority (1-10, higher = more important)
        enabled: Whether the SOP is active
    """
    name: str
    description: str
    trigger: str
    steps: List[SOPStep]
    category: str = "general"
    priority: int = 5
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the SOP to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "trigger": self.trigger,
            "category": self.category,
            "steps": [s.to_dict() for s in self.steps],
            "priority": self.priority,
            "enabled": self.enabled,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SOP":
        """Create a SOP from dictionary."""
        steps = [SOPStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            name=data["name"],
            description=data["description"],
            trigger=data["trigger"],
            category=data.get("category", "general"),
            steps=steps,
            priority=data.get("priority", 5),
            enabled=data.get("enabled", True),
            metadata=data.get("metadata", {}),
        )
    
    def to_prompt(self) -> str:
        """
        Generate a representation of the SOP for the prompt.
        
        Returns:
            Formatted string to include in the model's context
        """
        lines = [
            f"## Procedure: {self.name}",
            f"**Description**: {self.description}",
            f"**Category**: {self.category}",
            "",
            "### Steps:",
        ]
        
        for i, step in enumerate(self.steps, 1):
            step_line = f"{i}. {step.action}"
            if step.condition:
                step_line += f" (if: {step.condition})"
            if step.tools:
                step_line += f" [tools: {', '.join(step.tools)}]"
            lines.append(step_line)
        
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Reset all steps to pending."""
        for step in self.steps:
            step.status = StepStatus.PENDING
            step.result = None


# =============================================================================
# SOP MANAGER
# =============================================================================

class SOPManager:
    """
    Manager for SOPs (Standard Operating Procedures).
    
    Allows to:
    - Load/save SOPs from files
    - Find relevant SOPs for a query
    - Execute SOPs step-by-step
    - Generate context for the model
    """
    
    def __init__(self, sop_directory: Optional[str] = None):
        """
        Initialize the manager.
        
        Args:
            sop_directory: Directory to load/save SOPs (None = memory only)
        """
        self.sops: Dict[str, SOP] = {}
        self.sop_directory = sop_directory
        
        # Load SOPs from directory if specified
        if sop_directory and os.path.exists(sop_directory):
            self.load_sops_from_directory(sop_directory)
        
        # Add default SOPs
        self._add_default_sops()
    
    def _add_default_sops(self) -> None:
        """Add default SOPs for common tasks."""
        
        # SOP: Debug code
        self.add_sop(SOP(
            name="debug_python_code",
            description="Procedure for identifying and fixing bugs in Python code",
            trigger="debug, bug, error, not working, problem in code, fix",
            category="coding",
            priority=8,
            steps=[
                SOPStep(
                    action="Carefully read the code and the reported error",
                    expected_output="Context understanding",
                ),
                SOPStep(
                    action="Identify the error type (syntax, logic, runtime)",
                    expected_output="Error classification",
                ),
                SOPStep(
                    action="Locate the problematic line or function",
                    expected_output="Bug location",
                ),
                SOPStep(
                    action="Propose a solution with explanation",
                    expected_output="Corrected code + explanation",
                ),
                SOPStep(
                    action="Suggest tests to verify the fix",
                    expected_output="Test cases",
                ),
            ],
        ))
        
        # SOP: Write code
        self.add_sop(SOP(
            name="write_python_function",
            description="Procedure for writing a quality Python function",
            trigger="write, create, implement, function, code",
            category="coding",
            priority=7,
            steps=[
                SOPStep(
                    action="Understand the requirements and use cases",
                    expected_output="Requirements list",
                ),
                SOPStep(
                    action="Define the function signature with type hints",
                    expected_output="def function_name(params) -> ReturnType",
                ),
                SOPStep(
                    action="Write the docstring with description, args, returns",
                    expected_output="Complete docstring",
                ),
                SOPStep(
                    action="Implement the logic with error handling",
                    expected_output="Working code",
                ),
                SOPStep(
                    action="Add input validation if necessary",
                    condition="complex or user input",
                ),
                SOPStep(
                    action="Provide usage example",
                    expected_output="Example call",
                ),
            ],
        ))
        
        # SOP: RAG Search
        self.add_sop(SOP(
            name="rag_search_procedure",
            description="Procedure for answering using the knowledge base",
            trigger="search, find, knowledge base, documentation, RAG",
            category="rag",
            priority=9,
            steps=[
                SOPStep(
                    action="Identify keywords from the question",
                    expected_output="Keywords for search",
                ),
                SOPStep(
                    action="Search the knowledge base using search_knowledge_base",
                    tools=["search_knowledge_base"],
                    expected_output="Relevant documents",
                ),
                SOPStep(
                    action="Evaluate the relevance of results",
                    expected_output="Filtered results",
                ),
                SOPStep(
                    action="Synthesize the answer citing sources",
                    expected_output="Answer with citations",
                ),
                SOPStep(
                    action="If no information found, clearly admit it",
                    condition="no relevant results",
                    expected_output="Admission of missing info",
                ),
            ],
        ))
        
        # SOP: Code Review
        self.add_sop(SOP(
            name="code_review",
            description="Procedure for code review",
            trigger="review, check the code, feedback",
            category="coding",
            priority=7,
            steps=[
                SOPStep(
                    action="Verify logical correctness",
                    expected_output="List of logic issues",
                ),
                SOPStep(
                    action="Check style and conventions (PEP8)",
                    expected_output="Style issues",
                ),
                SOPStep(
                    action="Evaluate error handling",
                    expected_output="Error coverage",
                ),
                SOPStep(
                    action="Verify type hints and documentation",
                    expected_output="Docs completeness",
                ),
                SOPStep(
                    action="Suggest optimizations if appropriate",
                    condition="working but improvable code",
                ),
                SOPStep(
                    action="Provide constructive feedback",
                    expected_output="Review summary",
                ),
            ],
        ))
        
        # SOP: Explain concept
        self.add_sop(SOP(
            name="explain_concept",
            description="Procedure for explaining a technical concept",
            trigger="explain, what is, how does it work, what does it mean",
            category="education",
            priority=6,
            steps=[
                SOPStep(
                    action="Provide a simple definition (1-2 sentences)",
                    expected_output="Basic definition",
                ),
                SOPStep(
                    action="Explain the concept in detail",
                    expected_output="In-depth explanation",
                ),
                SOPStep(
                    action="Provide an analogy or practical example",
                    expected_output="Concrete example",
                ),
                SOPStep(
                    action="Show a code example if appropriate",
                    condition="programming concept",
                    expected_output="Example code",
                ),
                SOPStep(
                    action="Indicate resources for further learning",
                    expected_output="Links/references",
                ),
            ],
        ))
    
    def add_sop(self, sop: SOP) -> None:
        """Add a SOP to the manager."""
        self.sops[sop.name] = sop
        logger.debug(f"SOP added: {sop.name}")
    
    def remove_sop(self, name: str) -> bool:
        """Remove a SOP."""
        if name in self.sops:
            del self.sops[name]
            return True
        return False
    
    def get_sop(self, name: str) -> Optional[SOP]:
        """Get a SOP by name."""
        return self.sops.get(name)
    
    def list_sops(self, category: Optional[str] = None) -> List[SOP]:
        """
        List all SOPs.
        
        Args:
            category: Filter by category (None = all)
            
        Returns:
            List of SOPs sorted by priority
        """
        sops = list(self.sops.values())
        
        if category:
            sops = [s for s in sops if s.category == category]
        
        return sorted(sops, key=lambda x: -x.priority)
    
    def find_relevant_sop(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 1,
    ) -> List[SOP]:
        """
        Find the most relevant SOPs for a query.
        
        Uses simple matching on triggers. For semantic matching,
        integrate with VectorStore.
        
        Args:
            query: User's query
            category: Category filter
            top_k: Number of SOPs to return
            
        Returns:
            List of most relevant SOPs
        """
        query_lower = query.lower()
        scored_sops = []
        
        for sop in self.sops.values():
            if not sop.enabled:
                continue
            
            if category and sop.category != category:
                continue
            
            # Calculate score based on trigger match
            trigger_words = sop.trigger.lower().split(",")
            score = 0
            
            for trigger in trigger_words:
                trigger = trigger.strip()
                if trigger in query_lower:
                    score += 10
                # Partial match
                for word in trigger.split():
                    if word in query_lower:
                        score += 2
            
            # Boost for priority
            score += sop.priority * 0.5
            
            if score > 0:
                scored_sops.append((sop, score))
        
        # Sort by score
        scored_sops.sort(key=lambda x: -x[1])
        
        return [sop for sop, _ in scored_sops[:top_k]]
    
    def get_sop_context(self, query: str) -> str:
        """
        Generate the SOP context for the model's prompt.
        
        Args:
            query: User's query
            
        Returns:
            String with the SOP to follow (or empty if none)
        """
        relevant = self.find_relevant_sop(query, top_k=1)
        
        if not relevant:
            return ""
        
        sop = relevant[0]
        
        context = [
            "---",
            "**PROCEDURE TO FOLLOW:**",
            "",
            sop.to_prompt(),
            "",
            "Follow this procedure step-by-step. Indicate which step you are executing.",
            "---",
        ]
        
        return "\n".join(context)
    
    def save_sop(self, sop: SOP, filepath: Optional[str] = None) -> str:
        """
        Save a SOP to a JSON file.
        
        Args:
            sop: SOP to save
            filepath: File path (None = use sop_directory)
            
        Returns:
            Path of the saved file
        """
        if filepath is None:
            if self.sop_directory is None:
                raise ValueError("No SOP directory configured")
            os.makedirs(self.sop_directory, exist_ok=True)
            filepath = os.path.join(self.sop_directory, f"{sop.name}.json")
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(sop.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"SOP saved: {filepath}")
        return filepath
    
    def load_sop(self, filepath: str) -> SOP:
        """Load a SOP from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        sop = SOP.from_dict(data)
        self.add_sop(sop)
        return sop
    
    def load_sops_from_directory(self, directory: str) -> int:
        """
        Load all SOPs from a directory.
        
        Args:
            directory: Directory with .json files
            
        Returns:
            Number of SOPs loaded
        """
        count = 0
        path = Path(directory)
        
        for json_file in path.glob("*.json"):
            try:
                self.load_sop(str(json_file))
                count += 1
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
        
        logger.info(f"Loaded {count} SOPs from {directory}")
        return count
    
    def export_all(self, directory: str) -> int:
        """
        Export all SOPs to a directory.
        
        Args:
            directory: Destination directory
            
        Returns:
            Number of SOPs exported
        """
        os.makedirs(directory, exist_ok=True)
        count = 0
        
        for sop in self.sops.values():
            filepath = os.path.join(directory, f"{sop.name}.json")
            self.save_sop(sop, filepath)
            count += 1
        
        return count


# =============================================================================
# SYSTEM PROMPT WITH SOP
# =============================================================================

SYSTEM_PROMPT_WITH_SOP = """You are an AI assistant that follows Standard Operating Procedures (SOPs) when applicable.

When you identify that a task matches a known procedure:
1. State which procedure you're following
2. Execute each step in order
3. Report the result of each step
4. Skip steps with unmet conditions
5. Provide a summary at the end

{sop_context}

If no procedure is applicable, respond naturally and helpfully."""


def get_system_prompt_with_sop(query: str, sop_manager: SOPManager) -> str:
    """
    Generate the system prompt with the appropriate SOP.
    
    Args:
        query: User's query
        sop_manager: SOP manager
        
    Returns:
        Complete system prompt
    """
    sop_context = sop_manager.get_sop_context(query)
    return SYSTEM_PROMPT_WITH_SOP.format(sop_context=sop_context)
