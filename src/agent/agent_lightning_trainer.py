"""
Complete integration with Microsoft Agent Lightning.

Agent Lightning is the framework for training AI agents with:
- Reinforcement Learning (GRPO, PPO)
- Automatic Prompt Optimization (APO)
- Advanced Supervised Fine-Tuning (SFT)
- Span tracing for debugging

This module provides:
- AgentLightningTrainer: Complete wrapper for RL training
- RewardFunctions: Reward functions for coding, function calling, etc.
- SpanTracker: Detailed generation tracing

References:
- GitHub: https://github.com/microsoft/agent-lightning
- Docs: https://microsoft.github.io/agent-lightning/
"""

from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import re

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer, PreTrainedModel

# Agent Lightning imports
try:
    import agentlightning as agl
    from agentlightning import Trainer, LightningStore
    from agentlightning.algorithms import GRPO, SFT, APO
    from agentlightning.tracing import Span, Tracer
    AGENT_LIGHTNING_AVAILABLE = True
except ImportError:
    AGENT_LIGHTNING_AVAILABLE = False
    agl = None
    Trainer = None
    LightningStore = None
    GRPO = None
    SFT = None
    APO = None

logger = logging.getLogger(__name__)


class TrainingAlgorithm(str, Enum):
    """Training algorithms supported by Agent Lightning."""
    SFT = "sft"           # Supervised Fine-Tuning
    GRPO = "grpo"         # Group Relative Policy Optimization (RL)
    APO = "apo"           # Automatic Prompt Optimization


@dataclass
class AgentLightningConfig:
    """Configuration for Agent Lightning."""
    
    # Training algorithm
    algorithm: TrainingAlgorithm = TrainingAlgorithm.SFT
    
    # GRPO Configuration (Reinforcement Learning)
    grpo_config: Dict[str, Any] = field(default_factory=lambda: {
        "num_generations": 4,      # Generations per prompt
        "temperature": 0.7,        # Sampling temperature
        "top_p": 0.9,             # Nucleus sampling
        "max_new_tokens": 512,    # Max generated tokens
        "kl_coef": 0.1,           # KL divergence coefficient
        "gamma": 0.99,            # Discount factor
        "clip_range": 0.2,        # PPO clip range
    })
    
    # APO Configuration
    apo_config: Dict[str, Any] = field(default_factory=lambda: {
        "num_prompt_candidates": 5,
        "eval_samples": 20,
        "optimize_system_prompt": True,
    })
    
    # LightningStore
    store_path: str = "./lightning_store"
    enable_tracing: bool = True
    
    # Reward function to use
    reward_function: str = "combined"  # "coding", "function_calling", "combined"


# =============================================================================
# REWARD FUNCTIONS - The heart of RL training
# =============================================================================

class RewardFunction:
    """
    Reward functions for evaluating model generations.
    
    The reward guides the RL algorithm (GRPO) to improve agent behavior.
    """
    
    @staticmethod
    def coding_reward(
        prompt: str,
        generation: str,
        reference: Optional[str] = None,
    ) -> float:
        """
        Calculate reward for coding tasks.
        
        Criteria:
        - Correct syntax (try to parse)
        - Presence of docstring
        - Appropriate length
        - Match with reference (if available)
        
        Args:
            prompt: The original prompt
            generation: The generated response
            reference: Reference response (optional)
            
        Returns:
            Reward float between -1.0 and 1.0
        """
        reward = 0.0
        
        # 1. Extract code from generation
        code_blocks = re.findall(r'```(?:python)?\n?(.*?)```', generation, re.DOTALL)
        if not code_blocks:
            # If no code blocks but looks like code
            if 'def ' in generation or 'class ' in generation:
                code = generation
            else:
                return -0.5  # Penalize absence of code
        else:
            code = code_blocks[0]
        
        # 2. Verify Python syntax
        try:
            compile(code, '<string>', 'exec')
            reward += 0.3  # Correct syntax
        except SyntaxError:
            reward -= 0.3  # Syntax error
        
        # 3. Presence of docstring
        if '"""' in code or "'''" in code:
            reward += 0.1
        
        # 4. Presence of type hints
        if ': ' in code and '->' in code:
            reward += 0.1
        
        # 5. Appropriate length (not too short, not too long)
        code_len = len(code.strip())
        if 50 < code_len < 2000:
            reward += 0.1
        elif code_len < 20:
            reward -= 0.2  # Too short
        
        # 6. Compare with reference (if available)
        if reference:
            # Simple keyword overlap
            ref_keywords = set(re.findall(r'\b\w+\b', reference.lower()))
            gen_keywords = set(re.findall(r'\b\w+\b', code.lower()))
            overlap = len(ref_keywords & gen_keywords) / max(len(ref_keywords), 1)
            reward += 0.4 * overlap
        
        return max(-1.0, min(1.0, reward))
    
    @staticmethod
    def function_calling_reward(
        prompt: str,
        generation: str,
        available_tools: Optional[List[str]] = None,
        reference: Optional[str] = None,
    ) -> float:
        """
        Calculate reward for function calling tasks.
        
        Criteria:
        - Correct call format (valid JSON)
        - Existing tool (if list available)
        - Valid arguments
        - Match with reference
        
        Args:
            prompt: The prompt with the request
            generation: The response with function call
            available_tools: List of available tools
            reference: Reference call
            
        Returns:
            Reward float between -1.0 and 1.0
        """
        reward = 0.0
        
        # 1. Search for function call pattern
        # Supports various formats: <function_call>, {"name": ...}, tool_call, etc.
        fc_patterns = [
            r'<function_call>\s*(\{.*?\})\s*</function_call>',
            r'"function_call":\s*(\{.*?\})',
            r'```json\s*(\{.*?\})\s*```',
            r'\{"name":\s*"[^"]+",\s*"arguments":\s*\{.*?\}\}',
        ]
        
        function_call = None
        for pattern in fc_patterns:
            matches = re.findall(pattern, generation, re.DOTALL)
            if matches:
                function_call = matches[0]
                break
        
        if not function_call:
            # No function call found
            if "function" in prompt.lower() or "tool" in prompt.lower():
                return -0.5  # Should have called a function
            return 0.0  # Not necessary
        
        # 2. Verify valid JSON
        try:
            fc_data = json.loads(function_call) if isinstance(function_call, str) else function_call
            reward += 0.3  # Valid JSON
        except json.JSONDecodeError:
            return -0.3  # Invalid JSON
        
        # 3. Verify correct structure
        if "name" in fc_data:
            reward += 0.1
            
            # 4. Verify existing tool
            if available_tools:
                if fc_data["name"] in available_tools:
                    reward += 0.2
                else:
                    reward -= 0.2  # Non-existent tool
        
        if "arguments" in fc_data:
            reward += 0.1
            
            # Verify arguments is a dict
            if isinstance(fc_data["arguments"], dict):
                reward += 0.1
        
        # 5. Compare with reference
        if reference:
            try:
                ref_data = json.loads(reference)
                if fc_data.get("name") == ref_data.get("name"):
                    reward += 0.2
                    # Check arguments
                    if fc_data.get("arguments") == ref_data.get("arguments"):
                        reward += 0.2
            except (json.JSONDecodeError, AttributeError):
                pass
        
        return max(-1.0, min(1.0, reward))
    
    @staticmethod
    def chat_reward(
        prompt: str,
        generation: str,
        reference: Optional[str] = None,
    ) -> float:
        """
        Calculate reward for chat/conversation tasks.
        
        Criteria:
        - Relevant response (not empty, not too short)
        - Not repetitive
        - Coherent with prompt
        - Fluency
        
        Args:
            prompt: The question/instruction
            generation: The generated response
            reference: Reference response
            
        Returns:
            Reward float between -1.0 and 1.0
        """
        reward = 0.0
        
        # 1. Appropriate length
        gen_len = len(generation.strip())
        if gen_len < 10:
            return -0.5  # Too short
        elif gen_len > 50:
            reward += 0.1
        
        # 2. Not repetitive (penalize repetitions)
        words = generation.lower().split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                reward -= 0.3  # Very repetitive
            elif unique_ratio > 0.7:
                reward += 0.2
        
        # 3. Answers the question (keyword overlap)
        prompt_words = set(re.findall(r'\b\w+\b', prompt.lower()))
        gen_words = set(re.findall(r'\b\w+\b', generation.lower()))
        
        # Remove common words
        common_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", 
                       "being", "have", "has", "had", "do", "does", "did", "will",
                       "would", "could", "should", "may", "might", "must", "shall",
                       "to", "of", "in", "for", "on", "with", "at", "by", "from",
                       "as", "into", "through", "during", "before", "after", "above",
                       "below", "between", "under", "again", "further", "then", "once",
                       "here", "there", "when", "where", "why", "how", "all", "each",
                       "few", "more", "most", "other", "some", "such", "no", "nor",
                       "not", "only", "own", "same", "so", "than", "too", "very",
                       "can", "just", "should", "now", "i", "you", "he", "she", "it",
                       "we", "they", "what", "which", "who", "this", "that", "these",
                       "those", "am", "and", "but", "if", "or", "because", "until",
                       "while", "although", "though", "after", "before", "unless"}
        
        prompt_words -= common_words
        gen_words -= common_words
        
        if prompt_words:
            overlap = len(prompt_words & gen_words) / len(prompt_words)
            reward += 0.3 * overlap
        
        # 4. Compare with reference
        if reference:
            ref_words = set(re.findall(r'\b\w+\b', reference.lower())) - common_words
            if ref_words:
                ref_overlap = len(ref_words & gen_words) / len(ref_words)
                reward += 0.3 * ref_overlap
        
        return max(-1.0, min(1.0, reward))
    
    @staticmethod
    def combined_reward(
        prompt: str,
        generation: str,
        reference: Optional[str] = None,
        available_tools: Optional[List[str]] = None,
    ) -> float:
        """
        Combined reward that auto-detects task type.
        
        Examines the prompt to determine if it's:
        - Coding task
        - Function calling task
        - General chat
        
        Args:
            prompt: The prompt
            generation: Generated response
            reference: Reference (optional)
            available_tools: Available tools (optional)
            
        Returns:
            Appropriate reward for detected task type
        """
        prompt_lower = prompt.lower()
        
        # Detect function calling
        fc_keywords = ["function", "tool", "call", "api", "execute", "invoke"]
        if any(kw in prompt_lower for kw in fc_keywords):
            return RewardFunction.function_calling_reward(
                prompt, generation, available_tools, reference
            )
        
        # Detect coding
        coding_keywords = ["code", "python", "function", "class", "implement", 
                         "write", "program", "script", "def ", "import "]
        if any(kw in prompt_lower for kw in coding_keywords):
            return RewardFunction.coding_reward(prompt, generation, reference)
        
        # Default: chat
        return RewardFunction.chat_reward(prompt, generation, reference)


# =============================================================================
# AGENT LIGHTNING TRAINER
# =============================================================================

class AgentLightningTrainer:
    """
    Complete wrapper for training with Agent Lightning.
    
    Supports:
    - GRPO (Group Relative Policy Optimization) for RL
    - APO (Automatic Prompt Optimization)
    - SFT (Supervised Fine-Tuning)
    
    Example:
        ```python
        trainer = AgentLightningTrainer(model, tokenizer, config)
        results = trainer.train(train_dataset, num_epochs=3)
        ```
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: AgentLightningConfig,
        reward_fn: Optional[Callable] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model with LoRA/PEFT applied
            tokenizer: Tokenizer
            config: Agent Lightning configuration
            reward_fn: Custom reward function (optional)
        """
        if not AGENT_LIGHTNING_AVAILABLE:
            raise ImportError(
                "Agent Lightning not installed. Install with: pip install agentlightning"
            )
        
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Set reward function
        if reward_fn is not None:
            self.reward_fn = reward_fn
        else:
            self.reward_fn = self._get_default_reward_fn()
        
        # Initialize LightningStore
        self.store = LightningStore(path=config.store_path)
        
        # Initialize Tracer if enabled
        self.tracer = Tracer() if config.enable_tracing else None
        
        # Initialize algorithm
        self.algorithm = self._create_algorithm()
        
        # Agent Lightning Trainer
        self.trainer = None
        
        logger.info(f"AgentLightningTrainer initialized with algorithm: {config.algorithm.value}")
    
    def _get_default_reward_fn(self) -> Callable:
        """Return the default reward function based on config."""
        reward_type = self.config.reward_function
        
        if reward_type == "coding":
            return lambda p, g, r=None: RewardFunction.coding_reward(p, g, r)
        elif reward_type == "function_calling":
            return lambda p, g, r=None: RewardFunction.function_calling_reward(p, g, reference=r)
        else:  # combined
            return lambda p, g, r=None: RewardFunction.combined_reward(p, g, reference=r)
    
    def _create_algorithm(self):
        """Create the appropriate training algorithm."""
        if self.config.algorithm == TrainingAlgorithm.GRPO:
            return GRPO(
                model=self.model,
                tokenizer=self.tokenizer,
                reward_fn=self.reward_fn,
                **self.config.grpo_config,
            )
        elif self.config.algorithm == TrainingAlgorithm.APO:
            return APO(
                model=self.model,
                tokenizer=self.tokenizer,
                **self.config.apo_config,
            )
        else:  # SFT
            return SFT(
                model=self.model,
                tokenizer=self.tokenizer,
            )
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        output_dir: str = "./checkpoints",
    ) -> Dict[str, Any]:
        """
        Start training with Agent Lightning.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            num_epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            output_dir: Directory for checkpoints
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Starting training with {self.config.algorithm.value}")
        logger.info(f"  - Epochs: {num_epochs}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Learning rate: {learning_rate}")
        
        # Create Agent Lightning Trainer
        self.trainer = Trainer(
            algorithm=self.algorithm,
            store=self.store,
            output_dir=output_dir,
        )
        
        # Start training
        with self._trace_span("training", {"algorithm": self.config.algorithm.value}):
            results = self.trainer.train(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )
        
        # Save final model
        self.save_model(output_dir)
        
        return results
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        trace: bool = True,
    ) -> str:
        """
        Generate a response with optional tracing.
        
        Args:
            prompt: The input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            trace: Whether to trace the generation
            
        Returns:
            Generated text
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate with tracing
        with self._trace_span("generation", {"prompt_len": len(prompt)}) as span:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            
            if span:
                span.set_attribute("output_len", len(generated_text))
        
        # Emit event for tracing
        if self.config.enable_tracing:
            agl.emit_generation(
                prompt=prompt,
                response=generated_text,
                model=self.model.config._name_or_path,
            )
        
        return generated_text
    
    def evaluate_reward(
        self,
        prompt: str,
        generation: str,
        reference: Optional[str] = None,
    ) -> float:
        """
        Evaluate the reward for a generation.
        
        Args:
            prompt: Original prompt
            generation: Generated text
            reference: Optional reference
            
        Returns:
            Reward value
        """
        return self.reward_fn(prompt, generation, reference)
    
    def save_model(self, output_dir: str) -> None:
        """Save the model and LoRA adapter."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save LoRA adapter
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save Agent Lightning config
        config_path = os.path.join(output_dir, "agent_lightning_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "algorithm": self.config.algorithm.value,
                "grpo_config": self.config.grpo_config,
                "apo_config": self.config.apo_config,
                "reward_function": self.config.reward_function,
            }, f, indent=2)
        
        logger.info(f"Model saved to: {output_dir}")
    
    def _trace_span(self, name: str, attributes: Optional[Dict] = None):
        """Context manager for span tracing."""
        if self.tracer and self.config.enable_tracing:
            return self.tracer.span(name, attributes=attributes or {})
        
        # Dummy context manager if tracing disabled
        from contextlib import nullcontext
        return nullcontext()


# =============================================================================
# FACTORY AND UTILITIES
# =============================================================================

def create_agent_lightning_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: Dict[str, Any],
) -> AgentLightningTrainer:
    """
    Factory function to create AgentLightningTrainer from YAML config.
    
    Args:
        model: Model
        tokenizer: Tokenizer
        config: Complete config (from config.yaml)
        
    Returns:
        Configured AgentLightningTrainer
    """
    agl_config = config.get("agent_lightning", {})
    
    algorithm_str = agl_config.get("algorithm", "sft")
    try:
        algorithm = TrainingAlgorithm(algorithm_str.lower())
    except ValueError:
        logger.warning(f"Invalid algorithm '{algorithm_str}', using SFT")
        algorithm = TrainingAlgorithm.SFT
    
    trainer_config = AgentLightningConfig(
        algorithm=algorithm,
        grpo_config=agl_config.get("grpo", {}),
        apo_config=agl_config.get("apo", {}),
        store_path=agl_config.get("store_path", "./lightning_store"),
        enable_tracing=agl_config.get("enable_tracing", True),
        reward_function=agl_config.get("reward_function", "combined"),
    )
    
    return AgentLightningTrainer(
        model=model,
        tokenizer=tokenizer,
        config=trainer_config,
    )


def check_agent_lightning_available() -> bool:
    """Check if Agent Lightning is installed."""
    return AGENT_LIGHTNING_AVAILABLE
