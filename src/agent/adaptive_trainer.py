"""
Adaptive Trainer: Dynamic optimization during training.

Inspired by AgentFlow's "in-the-flow" optimization approach, this module
implements adaptive training strategies that:
- Monitor training metrics in real-time
- Automatically adjust hyperparameters (learning rate, temperature, etc.)
- Implement curriculum learning (progressive difficulty)
- Detect and respond to training pathologies (divergence, plateaus)

The key insight from AgentFlow is that agentic systems should optimize
themselves during execution, not just before training.

References:
- AgentFlow: https://github.com/lupantech/AgentFlow
- Concept: "In-the-flow" optimization for agentic systems
"""

from typing import Optional, Dict, Any, List, Callable, Tuple, Deque
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging
import math
import json
from datetime import datetime

import torch

logger = logging.getLogger(__name__)


class TrainingState(str, Enum):
    """Current state of the training process."""
    WARMING_UP = "warming_up"
    STABLE = "stable"
    IMPROVING = "improving"
    PLATEAUING = "plateauing"
    DIVERGING = "diverging"
    CONVERGED = "converged"


class AdaptiveAction(str, Enum):
    """Actions the adaptive trainer can take."""
    NO_ACTION = "no_action"
    REDUCE_LR = "reduce_lr"
    INCREASE_LR = "increase_lr"
    REDUCE_TEMPERATURE = "reduce_temperature"
    INCREASE_TEMPERATURE = "increase_temperature"
    INCREASE_BATCH_SIZE = "increase_batch_size"
    DECREASE_BATCH_SIZE = "decrease_batch_size"
    INCREASE_DIFFICULTY = "increase_difficulty"
    DECREASE_DIFFICULTY = "decrease_difficulty"
    EARLY_STOP = "early_stop"


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive training."""
    
    # Window sizes for metric analysis
    short_window: int = 10       # Recent steps for trend detection
    long_window: int = 50        # Historical context
    
    # Thresholds for state detection
    plateau_threshold: float = 0.001    # Min improvement to not be plateau
    divergence_threshold: float = 0.5   # Max loss increase before divergence
    improvement_threshold: float = 0.01 # Min improvement for "improving" state
    
    # Adaptation rates
    lr_reduction_factor: float = 0.5    # How much to reduce LR
    lr_increase_factor: float = 1.2     # How much to increase LR
    temperature_delta: float = 0.1      # Temperature adjustment step
    
    # Bounds
    min_lr: float = 1e-7
    max_lr: float = 1e-2
    min_temperature: float = 0.1
    max_temperature: float = 1.5
    
    # Curriculum learning
    enable_curriculum: bool = True
    initial_difficulty: float = 0.3     # Start with easier examples (0-1)
    max_difficulty: float = 1.0         # Maximum difficulty
    difficulty_increase_rate: float = 0.1  # How fast to increase
    
    # Early stopping
    patience: int = 10                  # Steps without improvement before action
    early_stop_patience: int = 30       # Steps before early stopping
    
    # Logging
    log_every_n_steps: int = 10
    save_adaptation_history: bool = True


@dataclass
class TrainingMetrics:
    """Snapshot of training metrics at a point in time."""
    step: int
    loss: float
    reward_mean: float = 0.0
    reward_std: float = 0.0
    learning_rate: float = 0.0
    temperature: float = 0.7
    difficulty: float = 1.0
    perplexity: float = 0.0
    gradient_norm: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "loss": self.loss,
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
            "difficulty": self.difficulty,
            "perplexity": self.perplexity,
            "gradient_norm": self.gradient_norm,
            "timestamp": self.timestamp,
        }


@dataclass
class AdaptationEvent:
    """Record of an adaptation action taken."""
    step: int
    state: TrainingState
    action: AdaptiveAction
    reason: str
    old_value: float
    new_value: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "state": self.state.value,
            "action": self.action.value,
            "reason": self.reason,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "timestamp": self.timestamp,
        }


# =============================================================================
# METRIC ANALYZER: Analyzes training dynamics
# =============================================================================

class MetricAnalyzer:
    """
    Analyzes training metrics to detect patterns and recommend actions.
    
    Uses statistical analysis of metric history to identify:
    - Trends (improving, degrading)
    - Plateaus (lack of progress)
    - Divergence (loss explosion)
    - Convergence (training complete)
    """
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.loss_history: Deque[float] = deque(maxlen=config.long_window)
        self.reward_history: Deque[float] = deque(maxlen=config.long_window)
        self.gradient_history: Deque[float] = deque(maxlen=config.short_window)
    
    def update(self, metrics: TrainingMetrics) -> None:
        """Update history with new metrics."""
        self.loss_history.append(metrics.loss)
        self.reward_history.append(metrics.reward_mean)
        self.gradient_history.append(metrics.gradient_norm)
    
    def get_trend(self, values: List[float], window: int) -> float:
        """
        Calculate the trend (slope) of a metric over a window.
        
        Returns positive for improvement (decreasing loss), negative for degradation.
        """
        if len(values) < 2:
            return 0.0
        
        window = min(window, len(values))
        recent = list(values)[-window:]
        
        # Simple linear regression slope
        n = len(recent)
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n
        
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        
        # Normalize by mean to get relative change
        if y_mean != 0:
            slope = slope / abs(y_mean)
        
        return -slope  # Negative slope (decreasing loss) is good
    
    def get_variance(self, values: List[float], window: int) -> float:
        """Calculate variance of recent values."""
        if len(values) < 2:
            return 0.0
        
        window = min(window, len(values))
        recent = list(values)[-window:]
        
        mean = sum(recent) / len(recent)
        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        
        return variance
    
    def detect_state(self) -> Tuple[TrainingState, str]:
        """
        Detect the current training state based on metric analysis.
        
        Returns:
            Tuple of (TrainingState, reason_string)
        """
        if len(self.loss_history) < self.config.short_window:
            return TrainingState.WARMING_UP, "Insufficient data for analysis"
        
        # Calculate trends
        short_trend = self.get_trend(list(self.loss_history), self.config.short_window)
        long_trend = self.get_trend(list(self.loss_history), self.config.long_window)
        
        # Calculate variance
        loss_variance = self.get_variance(list(self.loss_history), self.config.short_window)
        
        # Recent loss change
        recent_losses = list(self.loss_history)[-self.config.short_window:]
        loss_change = (recent_losses[0] - recent_losses[-1]) / max(recent_losses[0], 1e-8)
        
        # Detect divergence (loss increasing significantly)
        if short_trend < -self.config.divergence_threshold:
            return TrainingState.DIVERGING, f"Loss increasing rapidly (trend: {short_trend:.4f})"
        
        # Detect convergence (very low variance, minimal change)
        if loss_variance < self.config.plateau_threshold ** 2 and abs(short_trend) < self.config.plateau_threshold:
            recent = list(self.loss_history)[-5:]
            if max(recent) - min(recent) < self.config.plateau_threshold:
                return TrainingState.CONVERGED, "Loss converged with minimal variance"
        
        # Detect plateau
        if abs(short_trend) < self.config.plateau_threshold:
            return TrainingState.PLATEAUING, f"No significant progress (trend: {short_trend:.4f})"
        
        # Detect improvement
        if short_trend > self.config.improvement_threshold:
            return TrainingState.IMPROVING, f"Loss decreasing (trend: {short_trend:.4f})"
        
        return TrainingState.STABLE, "Training proceeding normally"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of current metric analysis."""
        if len(self.loss_history) < 2:
            return {"status": "insufficient_data"}
        
        return {
            "loss_current": self.loss_history[-1],
            "loss_mean": sum(self.loss_history) / len(self.loss_history),
            "loss_min": min(self.loss_history),
            "loss_trend_short": self.get_trend(list(self.loss_history), self.config.short_window),
            "loss_trend_long": self.get_trend(list(self.loss_history), self.config.long_window),
            "loss_variance": self.get_variance(list(self.loss_history), self.config.short_window),
            "reward_mean": sum(self.reward_history) / max(len(self.reward_history), 1),
        }


# =============================================================================
# CURRICULUM MANAGER: Manages progressive difficulty
# =============================================================================

class CurriculumManager:
    """
    Manages curriculum learning - progressive increase in task difficulty.
    
    Inspired by how humans learn: start with easy examples, gradually
    increase complexity as mastery is demonstrated.
    """
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.current_difficulty = config.initial_difficulty
        self.difficulty_history: List[Tuple[int, float]] = []
        self.steps_at_current_difficulty = 0
        self.performance_at_difficulty: Dict[float, List[float]] = {}
    
    def get_current_difficulty(self) -> float:
        """Get current difficulty level (0-1)."""
        return self.current_difficulty
    
    def update(self, step: int, reward: float) -> None:
        """Update curriculum state with new performance data."""
        self.steps_at_current_difficulty += 1
        
        # Track performance at this difficulty
        diff_key = round(self.current_difficulty, 2)
        if diff_key not in self.performance_at_difficulty:
            self.performance_at_difficulty[diff_key] = []
        self.performance_at_difficulty[diff_key].append(reward)
    
    def should_increase_difficulty(self) -> bool:
        """Check if we should increase difficulty."""
        if not self.config.enable_curriculum:
            return False
        
        if self.current_difficulty >= self.config.max_difficulty:
            return False
        
        # Check if we've mastered current difficulty
        diff_key = round(self.current_difficulty, 2)
        if diff_key not in self.performance_at_difficulty:
            return False
        
        recent_performance = self.performance_at_difficulty[diff_key][-10:]
        if len(recent_performance) < 10:
            return False
        
        # Mastery: high average reward with low variance
        avg_reward = sum(recent_performance) / len(recent_performance)
        variance = sum((r - avg_reward) ** 2 for r in recent_performance) / len(recent_performance)
        
        return avg_reward > 0.7 and variance < 0.1
    
    def should_decrease_difficulty(self) -> bool:
        """Check if we should decrease difficulty."""
        if not self.config.enable_curriculum:
            return False
        
        if self.current_difficulty <= self.config.initial_difficulty:
            return False
        
        # Check if we're struggling
        diff_key = round(self.current_difficulty, 2)
        if diff_key not in self.performance_at_difficulty:
            return False
        
        recent_performance = self.performance_at_difficulty[diff_key][-10:]
        if len(recent_performance) < 10:
            return False
        
        avg_reward = sum(recent_performance) / len(recent_performance)
        
        return avg_reward < 0.3
    
    def increase_difficulty(self, step: int) -> float:
        """Increase difficulty level."""
        old_difficulty = self.current_difficulty
        self.current_difficulty = min(
            self.config.max_difficulty,
            self.current_difficulty + self.config.difficulty_increase_rate
        )
        self.difficulty_history.append((step, self.current_difficulty))
        self.steps_at_current_difficulty = 0
        
        logger.info(f"Curriculum: Increased difficulty {old_difficulty:.2f} -> {self.current_difficulty:.2f}")
        
        return self.current_difficulty
    
    def decrease_difficulty(self, step: int) -> float:
        """Decrease difficulty level."""
        old_difficulty = self.current_difficulty
        self.current_difficulty = max(
            self.config.initial_difficulty,
            self.current_difficulty - self.config.difficulty_increase_rate
        )
        self.difficulty_history.append((step, self.current_difficulty))
        self.steps_at_current_difficulty = 0
        
        logger.info(f"Curriculum: Decreased difficulty {old_difficulty:.2f} -> {self.current_difficulty:.2f}")
        
        return self.current_difficulty
    
    def filter_by_difficulty(
        self,
        examples: List[Dict[str, Any]],
        difficulty_key: str = "difficulty",
    ) -> List[Dict[str, Any]]:
        """
        Filter examples by current difficulty level.
        
        Args:
            examples: List of examples with difficulty scores
            difficulty_key: Key in example dict containing difficulty
            
        Returns:
            Filtered list of appropriate examples
        """
        if not self.config.enable_curriculum:
            return examples
        
        margin = 0.2  # Include examples within this margin
        
        filtered = [
            ex for ex in examples
            if difficulty_key not in ex or 
            abs(ex[difficulty_key] - self.current_difficulty) <= margin
        ]
        
        # Ensure we have at least some examples
        if len(filtered) < len(examples) * 0.1:
            return examples
        
        return filtered


# =============================================================================
# ADAPTIVE TRAINER: Main orchestrator
# =============================================================================

class AdaptiveTrainer:
    """
    Adaptive Trainer: Dynamic optimization during training.
    
    Monitors training in real-time and automatically adjusts:
    - Learning rate (reduce on plateau, increase if stable)
    - Temperature (reduce for exploitation, increase for exploration)
    - Task difficulty (curriculum learning)
    - Batch size (for stability/speed tradeoff)
    
    Example:
        ```python
        adaptive = AdaptiveTrainer(config)
        
        for step, batch in enumerate(dataloader):
            # Normal training step
            loss, reward = train_step(batch)
            
            # Adaptive update
            actions = adaptive.step(
                step=step,
                loss=loss,
                reward_mean=reward.mean(),
                learning_rate=optimizer.param_groups[0]['lr'],
            )
            
            # Apply recommended actions
            for action in actions:
                if action.action == AdaptiveAction.REDUCE_LR:
                    optimizer.param_groups[0]['lr'] = action.new_value
        ```
    """
    
    def __init__(self, config: Optional[AdaptiveConfig] = None):
        """
        Initialize the Adaptive Trainer.
        
        Args:
            config: Adaptive training configuration
        """
        self.config = config or AdaptiveConfig()
        
        # Components
        self.analyzer = MetricAnalyzer(self.config)
        self.curriculum = CurriculumManager(self.config)
        
        # State
        self.current_state = TrainingState.WARMING_UP
        self.current_lr: float = 2e-4
        self.current_temperature: float = 0.7
        self.steps_without_improvement: int = 0
        self.best_loss: float = float('inf')
        
        # History
        self.metrics_history: List[TrainingMetrics] = []
        self.adaptation_history: List[AdaptationEvent] = []
        
        logger.info("AdaptiveTrainer initialized")
    
    def step(
        self,
        step: int,
        loss: float,
        reward_mean: float = 0.0,
        reward_std: float = 0.0,
        learning_rate: float = 0.0,
        temperature: float = 0.7,
        gradient_norm: float = 0.0,
    ) -> List[AdaptationEvent]:
        """
        Process one training step and return recommended actions.
        
        Args:
            step: Current training step
            loss: Current loss value
            reward_mean: Mean reward (for RL training)
            reward_std: Reward standard deviation
            learning_rate: Current learning rate
            temperature: Current sampling temperature
            gradient_norm: Gradient norm (for stability monitoring)
            
        Returns:
            List of AdaptationEvents with recommended actions
        """
        # Create metrics snapshot
        metrics = TrainingMetrics(
            step=step,
            loss=loss,
            reward_mean=reward_mean,
            reward_std=reward_std,
            learning_rate=learning_rate,
            temperature=temperature,
            difficulty=self.curriculum.get_current_difficulty(),
            perplexity=math.exp(min(loss, 10)),  # Cap to prevent overflow
            gradient_norm=gradient_norm,
        )
        
        # Update state
        self.current_lr = learning_rate or self.current_lr
        self.current_temperature = temperature
        
        # Update analyzers
        self.analyzer.update(metrics)
        self.curriculum.update(step, reward_mean)
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Track best loss
        if loss < self.best_loss:
            self.best_loss = loss
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1
        
        # Detect current state
        self.current_state, reason = self.analyzer.detect_state()
        
        # Determine actions
        actions = self._determine_actions(step, metrics, reason)
        
        # Log periodically
        if step % self.config.log_every_n_steps == 0:
            self._log_status(step, metrics)
        
        return actions
    
    def _determine_actions(
        self,
        step: int,
        metrics: TrainingMetrics,
        state_reason: str,
    ) -> List[AdaptationEvent]:
        """Determine what adaptive actions to take."""
        actions = []
        
        # State-based actions
        if self.current_state == TrainingState.DIVERGING:
            # Emergency: reduce learning rate
            action = self._create_lr_action(
                step, AdaptiveAction.REDUCE_LR,
                "Divergence detected - reducing LR",
                reduction_factor=0.5,
            )
            if action:
                actions.append(action)
            
            # Also reduce temperature for more conservative generation
            action = self._create_temperature_action(
                step, AdaptiveAction.REDUCE_TEMPERATURE,
                "Divergence - reducing exploration",
            )
            if action:
                actions.append(action)
        
        elif self.current_state == TrainingState.PLATEAUING:
            if self.steps_without_improvement > self.config.patience:
                # Try increasing LR to escape plateau
                action = self._create_lr_action(
                    step, AdaptiveAction.INCREASE_LR,
                    f"Plateau for {self.steps_without_improvement} steps - increasing LR",
                    increase_factor=1.5,
                )
                if action:
                    actions.append(action)
                
                # Increase exploration
                action = self._create_temperature_action(
                    step, AdaptiveAction.INCREASE_TEMPERATURE,
                    "Plateau - increasing exploration",
                )
                if action:
                    actions.append(action)
            
            if self.steps_without_improvement > self.config.early_stop_patience:
                actions.append(AdaptationEvent(
                    step=step,
                    state=self.current_state,
                    action=AdaptiveAction.EARLY_STOP,
                    reason=f"No improvement for {self.steps_without_improvement} steps",
                    old_value=0,
                    new_value=1,
                ))
        
        elif self.current_state == TrainingState.IMPROVING:
            # Good progress - consider increasing difficulty
            if self.curriculum.should_increase_difficulty():
                new_diff = self.curriculum.increase_difficulty(step)
                actions.append(AdaptationEvent(
                    step=step,
                    state=self.current_state,
                    action=AdaptiveAction.INCREASE_DIFFICULTY,
                    reason="Mastered current difficulty level",
                    old_value=self.curriculum.current_difficulty - self.config.difficulty_increase_rate,
                    new_value=new_diff,
                ))
        
        elif self.current_state == TrainingState.STABLE:
            # Check curriculum adjustments
            if self.curriculum.should_decrease_difficulty():
                new_diff = self.curriculum.decrease_difficulty(step)
                actions.append(AdaptationEvent(
                    step=step,
                    state=self.current_state,
                    action=AdaptiveAction.DECREASE_DIFFICULTY,
                    reason="Struggling with current difficulty",
                    old_value=self.curriculum.current_difficulty + self.config.difficulty_increase_rate,
                    new_value=new_diff,
                ))
        
        # Store actions
        for action in actions:
            self.adaptation_history.append(action)
        
        return actions
    
    def _create_lr_action(
        self,
        step: int,
        action_type: AdaptiveAction,
        reason: str,
        reduction_factor: float = None,
        increase_factor: float = None,
    ) -> Optional[AdaptationEvent]:
        """Create a learning rate adjustment action."""
        old_lr = self.current_lr
        
        if action_type == AdaptiveAction.REDUCE_LR:
            factor = reduction_factor or self.config.lr_reduction_factor
            new_lr = max(self.config.min_lr, old_lr * factor)
        else:
            factor = increase_factor or self.config.lr_increase_factor
            new_lr = min(self.config.max_lr, old_lr * factor)
        
        if new_lr == old_lr:
            return None
        
        self.current_lr = new_lr
        
        return AdaptationEvent(
            step=step,
            state=self.current_state,
            action=action_type,
            reason=reason,
            old_value=old_lr,
            new_value=new_lr,
        )
    
    def _create_temperature_action(
        self,
        step: int,
        action_type: AdaptiveAction,
        reason: str,
    ) -> Optional[AdaptationEvent]:
        """Create a temperature adjustment action."""
        old_temp = self.current_temperature
        
        if action_type == AdaptiveAction.REDUCE_TEMPERATURE:
            new_temp = max(self.config.min_temperature, 
                          old_temp - self.config.temperature_delta)
        else:
            new_temp = min(self.config.max_temperature,
                          old_temp + self.config.temperature_delta)
        
        if new_temp == old_temp:
            return None
        
        self.current_temperature = new_temp
        
        return AdaptationEvent(
            step=step,
            state=self.current_state,
            action=action_type,
            reason=reason,
            old_value=old_temp,
            new_value=new_temp,
        )
    
    def _log_status(self, step: int, metrics: TrainingMetrics) -> None:
        """Log current training status."""
        summary = self.analyzer.get_summary()
        
        logger.info(
            f"Step {step} | State: {self.current_state.value} | "
            f"Loss: {metrics.loss:.4f} | LR: {self.current_lr:.2e} | "
            f"Temp: {self.current_temperature:.2f} | "
            f"Difficulty: {self.curriculum.get_current_difficulty():.2f}"
        )
    
    def get_recommended_lr(self) -> float:
        """Get the recommended learning rate."""
        return self.current_lr
    
    def get_recommended_temperature(self) -> float:
        """Get the recommended temperature."""
        return self.current_temperature
    
    def get_current_difficulty(self) -> float:
        """Get the current curriculum difficulty."""
        return self.curriculum.get_current_difficulty()
    
    def should_stop(self) -> bool:
        """Check if early stopping is recommended."""
        return any(
            a.action == AdaptiveAction.EARLY_STOP 
            for a in self.adaptation_history[-5:]
        )
    
    def get_state(self) -> TrainingState:
        """Get current training state."""
        return self.current_state
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of adaptive training state."""
        return {
            "current_state": self.current_state.value,
            "steps_without_improvement": self.steps_without_improvement,
            "best_loss": self.best_loss,
            "current_lr": self.current_lr,
            "current_temperature": self.current_temperature,
            "current_difficulty": self.curriculum.get_current_difficulty(),
            "total_adaptations": len(self.adaptation_history),
            "metrics_summary": self.analyzer.get_summary(),
        }
    
    def export_history(self, filepath: str) -> None:
        """Export training and adaptation history to JSON."""
        data = {
            "config": {
                "short_window": self.config.short_window,
                "long_window": self.config.long_window,
                "enable_curriculum": self.config.enable_curriculum,
            },
            "metrics": [m.to_dict() for m in self.metrics_history],
            "adaptations": [a.to_dict() for a in self.adaptation_history],
            "summary": self.get_summary(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported adaptive training history to: {filepath}")


# =============================================================================
# INTEGRATION WITH EXISTING TRAINERS
# =============================================================================

class AdaptiveTrainingCallback:
    """
    Callback for integrating AdaptiveTrainer with existing training loops.
    
    Can be used with PyTorch Lightning or custom training loops.
    """
    
    def __init__(self, config: Optional[AdaptiveConfig] = None):
        self.adaptive = AdaptiveTrainer(config)
        self.optimizer = None
    
    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Set the optimizer for automatic LR updates."""
        self.optimizer = optimizer
    
    def on_train_batch_end(
        self,
        step: int,
        loss: float,
        reward_mean: float = 0.0,
        **kwargs,
    ) -> List[AdaptationEvent]:
        """
        Called at the end of each training batch.
        
        Automatically applies LR changes if optimizer is set.
        """
        lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
        
        actions = self.adaptive.step(
            step=step,
            loss=loss,
            reward_mean=reward_mean,
            learning_rate=lr,
            **kwargs,
        )
        
        # Auto-apply LR changes
        if self.optimizer:
            for action in actions:
                if action.action in [AdaptiveAction.REDUCE_LR, AdaptiveAction.INCREASE_LR]:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = action.new_value
                    logger.info(f"Applied LR change: {action.old_value:.2e} -> {action.new_value:.2e}")
        
        return actions
    
    def should_stop(self) -> bool:
        """Check if training should stop."""
        return self.adaptive.should_stop()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_adaptive_trainer(
    enable_curriculum: bool = True,
    patience: int = 10,
    **kwargs,
) -> AdaptiveTrainer:
    """
    Factory function to create an AdaptiveTrainer.
    
    Args:
        enable_curriculum: Whether to enable curriculum learning
        patience: Steps without improvement before action
        **kwargs: Additional config parameters
        
    Returns:
        Configured AdaptiveTrainer
    """
    config = AdaptiveConfig(
        enable_curriculum=enable_curriculum,
        patience=patience,
        **kwargs,
    )
    return AdaptiveTrainer(config)


def create_adaptive_callback(
    optimizer: Optional[torch.optim.Optimizer] = None,
    **kwargs,
) -> AdaptiveTrainingCallback:
    """
    Factory function to create an AdaptiveTrainingCallback.
    
    Args:
        optimizer: Optional optimizer for auto LR updates
        **kwargs: Config parameters
        
    Returns:
        Configured callback
    """
    callback = AdaptiveTrainingCallback(AdaptiveConfig(**kwargs))
    if optimizer:
        callback.set_optimizer(optimizer)
    return callback

