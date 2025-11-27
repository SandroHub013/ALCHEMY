"""
Unsloth RL Trainer - High-performance Reinforcement Learning with Unsloth.

This module provides optimized RL training using Unsloth's performance
improvements combined with TRL's training algorithms:
- GRPO (Group Relative Policy Optimization)
- DPO (Direct Preference Optimization)
- ORPO, KTO, SimPO, and more

Unsloth provides 2x faster training and 70% less VRAM for RL methods.

References:
- Unsloth: https://github.com/unslothai/unsloth
- TRL: https://github.com/huggingface/trl
- GRPO Paper: https://arxiv.org/abs/2402.03300

Requirements:
    pip install unsloth trl
"""

from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import os

logger = logging.getLogger(__name__)

# Check dependencies
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    FastLanguageModel = None

try:
    from trl import (
        SFTTrainer,
        SFTConfig,
        GRPOTrainer,
        GRPOConfig,
        DPOTrainer,
        DPOConfig,
        ORPOTrainer,
        ORPOConfig,
    )
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    SFTTrainer = None
    GRPOTrainer = None
    DPOTrainer = None
    ORPOTrainer = None


class UnslothRLAlgorithm(str, Enum):
    """Supported RL algorithms with Unsloth."""
    SFT = "sft"       # Supervised Fine-Tuning
    GRPO = "grpo"     # Group Relative Policy Optimization
    DPO = "dpo"       # Direct Preference Optimization
    ORPO = "orpo"     # Odds Ratio Preference Optimization
    KTO = "kto"       # Kahneman-Tversky Optimization
    SIMPO = "simpo"   # Simple Preference Optimization


@dataclass
class UnslothTrainerConfig:
    """Configuration for Unsloth RL training."""
    
    # Algorithm selection
    algorithm: UnslothRLAlgorithm = UnslothRLAlgorithm.GRPO
    
    # Training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    max_steps: int = -1  # -1 means use epochs
    
    # Optimizer (adamw_8bit recommended for memory efficiency)
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    
    # Sequence length
    max_seq_length: int = 2048
    
    # GRPO-specific parameters
    grpo_num_generations: int = 4
    grpo_temperature: float = 0.7
    grpo_top_p: float = 0.95
    grpo_max_new_tokens: int = 512
    grpo_kl_coef: float = 0.1
    
    # DPO-specific parameters
    dpo_beta: float = 0.1
    
    # Logging
    logging_steps: int = 1
    save_steps: int = 100
    save_strategy: str = "steps"
    
    # Output
    output_dir: str = "./outputs"
    
    # Random seed
    seed: int = 3407
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True  # Recommended if GPU supports it
    
    # Report to
    report_to: str = "none"  # "none", "wandb", "tensorboard"


# =============================================================================
# REWARD FUNCTIONS FOR RL
# =============================================================================

class UnslothRewardFunctions:
    """
    Reward functions optimized for use with Unsloth RL training.
    
    These can be passed to GRPO trainer as reward_funcs.
    """
    
    @staticmethod
    def correctness_reward(
        prompts: List[str],
        completions: List[str],
        references: Optional[List[str]] = None,
    ) -> List[float]:
        """
        Basic correctness reward based on keyword overlap.
        
        Args:
            prompts: List of prompts
            completions: Model completions
            references: Optional reference answers
            
        Returns:
            List of rewards
        """
        import re
        rewards = []
        
        for i, completion in enumerate(completions):
            reward = 0.0
            
            # Length check
            if len(completion.strip()) < 10:
                rewards.append(-0.5)
                continue
            
            # Reference comparison if available
            if references and i < len(references):
                ref = references[i]
                ref_words = set(re.findall(r'\b\w+\b', ref.lower()))
                comp_words = set(re.findall(r'\b\w+\b', completion.lower()))
                
                if ref_words:
                    overlap = len(ref_words & comp_words) / len(ref_words)
                    reward = overlap
            else:
                # Basic quality heuristics
                reward = min(len(completion) / 500, 1.0) * 0.5
            
            rewards.append(max(-1.0, min(1.0, reward)))
        
        return rewards
    
    @staticmethod
    def coding_reward(
        prompts: List[str],
        completions: List[str],
        **kwargs,
    ) -> List[float]:
        """
        Reward function for coding tasks.
        
        Checks:
        - Valid Python syntax
        - Presence of docstrings
        - Type hints
        """
        import re
        rewards = []
        
        for completion in completions:
            reward = 0.0
            
            # Extract code blocks
            code_blocks = re.findall(r'```(?:python)?\n?(.*?)```', completion, re.DOTALL)
            code = code_blocks[0] if code_blocks else completion
            
            # Syntax check
            try:
                compile(code, '<string>', 'exec')
                reward += 0.4
            except SyntaxError:
                reward -= 0.3
            
            # Docstring bonus
            if '"""' in code or "'''" in code:
                reward += 0.2
            
            # Type hints bonus
            if '->' in code and ':' in code:
                reward += 0.2
            
            # Length penalty for very short code
            if len(code.strip()) < 20:
                reward -= 0.2
            
            rewards.append(max(-1.0, min(1.0, reward)))
        
        return rewards
    
    @staticmethod
    def reasoning_reward(
        prompts: List[str],
        completions: List[str],
        **kwargs,
    ) -> List[float]:
        """
        Reward function for reasoning tasks.
        
        Checks for step-by-step reasoning patterns.
        """
        import re
        rewards = []
        
        step_patterns = [
            r'step \d',
            r'\d\.',
            r'first[,:]',
            r'then[,:]',
            r'finally[,:]',
            r'therefore[,:]',
            r'because',
            r'since',
            r'thus',
        ]
        
        for completion in completions:
            reward = 0.0
            completion_lower = completion.lower()
            
            # Check for reasoning patterns
            pattern_count = sum(
                1 for p in step_patterns 
                if re.search(p, completion_lower)
            )
            
            # More patterns = better reasoning
            reward += min(pattern_count * 0.1, 0.5)
            
            # Length bonus (reasoning should be detailed)
            if len(completion) > 200:
                reward += 0.2
            elif len(completion) > 100:
                reward += 0.1
            
            # Conclusion indicator
            if any(w in completion_lower for w in ['therefore', 'thus', 'answer is', 'result is']):
                reward += 0.2
            
            rewards.append(max(-1.0, min(1.0, reward)))
        
        return rewards
    
    @staticmethod
    def create_combined_reward(
        coding_weight: float = 0.3,
        reasoning_weight: float = 0.3,
        correctness_weight: float = 0.4,
    ) -> Callable:
        """
        Create a combined reward function.
        
        Args:
            coding_weight: Weight for coding rewards
            reasoning_weight: Weight for reasoning rewards
            correctness_weight: Weight for correctness rewards
            
        Returns:
            Combined reward function
        """
        def combined_reward(
            prompts: List[str],
            completions: List[str],
            references: Optional[List[str]] = None,
        ) -> List[float]:
            coding_rewards = UnslothRewardFunctions.coding_reward(prompts, completions)
            reasoning_rewards = UnslothRewardFunctions.reasoning_reward(prompts, completions)
            correctness_rewards = UnslothRewardFunctions.correctness_reward(
                prompts, completions, references
            )
            
            combined = []
            for i in range(len(completions)):
                r = (
                    coding_weight * coding_rewards[i] +
                    reasoning_weight * reasoning_rewards[i] +
                    correctness_weight * correctness_rewards[i]
                )
                combined.append(r)
            
            return combined
        
        return combined_reward


# =============================================================================
# UNSLOTH RL TRAINER
# =============================================================================

class UnslothRLTrainer:
    """
    High-performance RL trainer using Unsloth + TRL.
    
    Combines Unsloth's memory and speed optimizations with TRL's
    RL algorithms (GRPO, DPO, etc.) for efficient training.
    
    Example:
        ```python
        from src.models import UnslothModelLoader
        from src.agent import UnslothRLTrainer
        
        # Load model with Unsloth
        loader = UnslothModelLoader("mistral-7b")
        model, tokenizer = loader.load()
        
        # Create trainer
        trainer = UnslothRLTrainer(
            model=model,
            tokenizer=tokenizer,
            algorithm="grpo",
            reward_fn=UnslothRewardFunctions.coding_reward,
        )
        
        # Train
        trainer.train(dataset)
        ```
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[UnslothTrainerConfig] = None,
        reward_fn: Optional[Callable] = None,
        reward_funcs: Optional[List[Callable]] = None,
    ):
        """
        Initialize the Unsloth RL trainer.
        
        Args:
            model: Model loaded with UnslothModelLoader
            tokenizer: Tokenizer
            config: Training configuration
            reward_fn: Single reward function (for GRPO)
            reward_funcs: List of reward functions (for GRPO)
        """
        if not UNSLOTH_AVAILABLE:
            raise ImportError("Unsloth not installed. pip install unsloth")
        
        if not TRL_AVAILABLE:
            raise ImportError("TRL not installed. pip install trl")
        
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or UnslothTrainerConfig()
        
        # Set up reward functions
        if reward_funcs:
            self.reward_funcs = reward_funcs
        elif reward_fn:
            self.reward_funcs = [reward_fn]
        else:
            # Default: combined reward
            self.reward_funcs = [UnslothRewardFunctions.create_combined_reward()]
        
        self.trainer = None
        
        logger.info(f"UnslothRLTrainer initialized with algorithm: {self.config.algorithm.value}")
    
    def _create_sft_trainer(self, dataset: Any) -> SFTTrainer:
        """Create SFT trainer."""
        training_args = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps if self.config.max_steps > 0 else None,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_strategy=self.config.save_strategy,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            seed=self.config.seed,
            max_seq_length=self.config.max_seq_length,
            report_to=self.config.report_to,
        )
        
        return SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=training_args,
        )
    
    def _create_grpo_trainer(self, dataset: Any) -> "GRPOTrainer":
        """Create GRPO trainer."""
        training_args = GRPOConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps if self.config.max_steps > 0 else None,
            optim=self.config.optim,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            seed=self.config.seed,
            max_completion_length=self.config.grpo_max_new_tokens,
            num_generations=self.config.grpo_num_generations,
            temperature=self.config.grpo_temperature,
            report_to=self.config.report_to,
        )
        
        return GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=dataset,
            reward_funcs=self.reward_funcs,
            args=training_args,
        )
    
    def _create_dpo_trainer(self, dataset: Any) -> "DPOTrainer":
        """Create DPO trainer."""
        training_args = DPOConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            optim=self.config.optim,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            seed=self.config.seed,
            beta=self.config.dpo_beta,
            max_length=self.config.max_seq_length,
            report_to=self.config.report_to,
        )
        
        return DPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=training_args,
        )
    
    def _create_orpo_trainer(self, dataset: Any) -> "ORPOTrainer":
        """Create ORPO trainer."""
        training_args = ORPOConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            optim=self.config.optim,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            seed=self.config.seed,
            max_length=self.config.max_seq_length,
            report_to=self.config.report_to,
        )
        
        return ORPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=training_args,
        )
    
    def create_trainer(self, dataset: Any) -> Any:
        """
        Create the appropriate trainer based on algorithm.
        
        Args:
            dataset: Training dataset
            
        Returns:
            Configured trainer
        """
        algorithm = self.config.algorithm
        
        if algorithm == UnslothRLAlgorithm.SFT:
            self.trainer = self._create_sft_trainer(dataset)
        elif algorithm == UnslothRLAlgorithm.GRPO:
            self.trainer = self._create_grpo_trainer(dataset)
        elif algorithm == UnslothRLAlgorithm.DPO:
            self.trainer = self._create_dpo_trainer(dataset)
        elif algorithm == UnslothRLAlgorithm.ORPO:
            self.trainer = self._create_orpo_trainer(dataset)
        else:
            raise ValueError(f"Algorithm {algorithm} not yet implemented")
        
        logger.info(f"Created {algorithm.value.upper()} trainer")
        
        return self.trainer
    
    def train(self, dataset: Any) -> Dict[str, Any]:
        """
        Run training with the configured algorithm.
        
        Args:
            dataset: Training dataset
            
        Returns:
            Training results/metrics
        """
        if self.trainer is None:
            self.create_trainer(dataset)
        
        logger.info(f"Starting training with {self.config.algorithm.value.upper()}")
        logger.info(f"  Epochs: {self.config.num_train_epochs}")
        logger.info(f"  Batch size: {self.config.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        
        # Train
        results = self.trainer.train()
        
        logger.info("Training complete!")
        
        return results
    
    def save(
        self,
        output_dir: Optional[str] = None,
        save_method: str = "lora",
    ) -> str:
        """
        Save the trained model.
        
        Args:
            output_dir: Directory to save (uses config default if None)
            save_method: "lora", "merged_16bit", "merged_4bit", "gguf"
            
        Returns:
            Path to saved model
        """
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if save_method == "lora":
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        elif save_method == "merged_16bit":
            self.model.save_pretrained_merged(
                output_dir, self.tokenizer, save_method="merged_16bit"
            )
        elif save_method == "merged_4bit":
            self.model.save_pretrained_merged(
                output_dir, self.tokenizer, save_method="merged_4bit"
            )
        elif save_method == "gguf":
            self.model.save_pretrained_gguf(
                output_dir, self.tokenizer, quantization_method="q4_k_m"
            )
        
        logger.info(f"Model saved to: {output_dir}")
        return output_dir
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a response (for testing).
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        # Enable inference mode for speed
        FastLanguageModel.for_inference(self.model)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
        
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        # Return to training mode
        FastLanguageModel.for_training(self.model)
        
        return generated


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_unsloth_rl_trainer(
    model: Any,
    tokenizer: Any,
    algorithm: str = "grpo",
    reward_fn: Optional[Callable] = None,
    **kwargs,
) -> UnslothRLTrainer:
    """
    Factory function to create an UnslothRLTrainer.
    
    Args:
        model: Model from UnslothModelLoader
        tokenizer: Tokenizer
        algorithm: "sft", "grpo", "dpo", "orpo"
        reward_fn: Reward function
        **kwargs: Additional config parameters
        
    Returns:
        Configured UnslothRLTrainer
    """
    algorithm_enum = UnslothRLAlgorithm(algorithm.lower())
    
    config = UnslothTrainerConfig(
        algorithm=algorithm_enum,
        **kwargs,
    )
    
    return UnslothRLTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        reward_fn=reward_fn,
    )


def quick_train_with_unsloth(
    model_name: str,
    dataset: Any,
    algorithm: str = "grpo",
    output_dir: str = "./outputs",
    **kwargs,
) -> Dict[str, Any]:
    """
    Quick helper to train a model with Unsloth.
    
    Args:
        model_name: Model name or alias
        dataset: Training dataset
        algorithm: RL algorithm
        output_dir: Output directory
        **kwargs: Additional parameters
        
    Returns:
        Training results
    """
    from .unsloth_loader import create_unsloth_loader
    
    # Load model
    loader = create_unsloth_loader(model_name)
    model, tokenizer = loader.load()
    
    # Create trainer
    trainer = create_unsloth_rl_trainer(
        model=model,
        tokenizer=tokenizer,
        algorithm=algorithm,
        output_dir=output_dir,
        **kwargs,
    )
    
    # Train
    results = trainer.train(dataset)
    
    # Save
    trainer.save(output_dir)
    
    return results

