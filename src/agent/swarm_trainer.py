"""
Swarm Trainer: Multi-agent orchestration for distributed training.

Inspired by claude-flow's swarm intelligence architecture, this module
implements a multi-agent training system that:
- Spawns multiple exploration agents in parallel
- Coordinates reward exploration across agents
- Aggregates best trajectories for policy updates
- Enables emergent collaborative behaviors

The key insight from claude-flow is that swarms of agents can explore
the solution space more efficiently than single agents.

References:
- claude-flow: https://github.com/ruvnet/claude-flow
- Concept: "Swarm intelligence" for AI agent coordination
"""

from typing import Optional, Dict, Any, List, Callable, Tuple, TypeVar
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import threading
import queue
import time
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import copy

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SwarmRole(str, Enum):
    """Roles that agents can play in the swarm."""
    EXPLORER = "explorer"       # Explores new solutions
    EXPLOITER = "exploiter"     # Exploits known good solutions
    SCOUT = "scout"             # Quick evaluation of many options
    SPECIALIST = "specialist"   # Deep dive into specific areas
    COORDINATOR = "coordinator" # Orchestrates other agents


class AgentStatus(str, Enum):
    """Status of an agent in the swarm."""
    IDLE = "idle"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SwarmConfig:
    """Configuration for the swarm trainer."""
    
    # Swarm size
    num_agents: int = 4                    # Number of parallel agents
    num_explorers: int = 2                 # Exploratory agents (high temp)
    num_exploiters: int = 2                # Exploitative agents (low temp)
    
    # Generation parameters
    explorer_temperature: float = 0.9      # High temp for exploration
    exploiter_temperature: float = 0.3     # Low temp for exploitation
    max_new_tokens: int = 512
    
    # Aggregation
    top_k_trajectories: int = 4            # Best trajectories to keep
    aggregation_method: str = "best"       # "best", "weighted", "ensemble"
    
    # Coordination
    sync_every_n_steps: int = 10           # How often to sync agents
    communication_enabled: bool = True      # Allow inter-agent communication
    
    # Diversity
    diversity_bonus: float = 0.1           # Bonus for diverse solutions
    min_diversity_threshold: float = 0.3   # Minimum diversity required
    
    # Resource management
    max_workers: int = 4                   # Thread pool size
    timeout_seconds: int = 60              # Timeout per agent task
    
    # Logging
    log_agent_outputs: bool = True
    save_all_trajectories: bool = False


@dataclass
class Trajectory:
    """A trajectory (prompt + generation + reward) from an agent."""
    agent_id: str
    prompt: str
    generation: str
    reward: float
    role: SwarmRole
    temperature: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "prompt": self.prompt,
            "generation": self.generation,
            "reward": self.reward,
            "role": self.role.value,
            "temperature": self.temperature,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class SwarmMessage:
    """Message for inter-agent communication."""
    sender_id: str
    content: str
    message_type: str  # "discovery", "warning", "suggestion"
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# SWARM AGENT: Individual agent in the swarm
# =============================================================================

class SwarmAgent:
    """
    Individual agent that operates as part of a swarm.
    
    Each agent has a specific role and configuration, and can
    communicate with other agents through the message bus.
    """
    
    def __init__(
        self,
        agent_id: str,
        role: SwarmRole,
        model: Any,
        tokenizer: Any,
        reward_fn: Callable[[str, str], float],
        config: SwarmConfig,
    ):
        """
        Initialize a swarm agent.
        
        Args:
            agent_id: Unique identifier
            role: Role in the swarm
            model: LLM model for generation
            tokenizer: Tokenizer
            reward_fn: Reward function
            config: Swarm configuration
        """
        self.agent_id = agent_id
        self.role = role
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config
        
        # Set temperature based on role
        if role == SwarmRole.EXPLORER:
            self.temperature = config.explorer_temperature
        elif role == SwarmRole.EXPLOITER:
            self.temperature = config.exploiter_temperature
        elif role == SwarmRole.SCOUT:
            self.temperature = 1.0  # Maximum exploration
        else:
            self.temperature = 0.7  # Default
        
        # State
        self.status = AgentStatus.IDLE
        self.trajectories: List[Trajectory] = []
        self.inbox: queue.Queue = queue.Queue()
        
        # Best solution found by this agent
        self.best_trajectory: Optional[Trajectory] = None
        
        logger.debug(f"SwarmAgent {agent_id} initialized with role {role.value}")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a response to a prompt.
        
        Args:
            prompt: Input prompt
            temperature: Override temperature
            
        Returns:
            Generated text
        """
        temp = temperature or self.temperature
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with self._no_grad_context():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=temp,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return generated
    
    def _no_grad_context(self):
        """Context manager for no gradient computation."""
        try:
            import torch
            return torch.no_grad()
        except ImportError:
            from contextlib import nullcontext
            return nullcontext()
    
    def explore(self, prompt: str) -> Trajectory:
        """
        Explore a prompt and return a trajectory.
        
        Args:
            prompt: Input prompt to explore
            
        Returns:
            Trajectory with generation and reward
        """
        self.status = AgentStatus.WORKING
        
        try:
            # Generate response
            generation = self.generate(prompt)
            
            # Calculate reward
            reward = self.reward_fn(prompt, generation)
            
            # Create trajectory
            trajectory = Trajectory(
                agent_id=self.agent_id,
                prompt=prompt,
                generation=generation,
                reward=reward,
                role=self.role,
                temperature=self.temperature,
            )
            
            # Store trajectory
            self.trajectories.append(trajectory)
            
            # Update best
            if self.best_trajectory is None or reward > self.best_trajectory.reward:
                self.best_trajectory = trajectory
            
            self.status = AgentStatus.COMPLETED
            
            return trajectory
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} exploration failed: {e}")
            self.status = AgentStatus.FAILED
            raise
    
    def multi_explore(self, prompt: str, n_samples: int = 3) -> List[Trajectory]:
        """
        Generate multiple explorations for a prompt.
        
        Args:
            prompt: Input prompt
            n_samples: Number of samples to generate
            
        Returns:
            List of trajectories
        """
        trajectories = []
        
        for _ in range(n_samples):
            try:
                traj = self.explore(prompt)
                trajectories.append(traj)
            except Exception:
                continue
        
        return trajectories
    
    def receive_message(self, message: SwarmMessage) -> None:
        """Receive a message from another agent."""
        self.inbox.put(message)
    
    def process_messages(self) -> List[SwarmMessage]:
        """Process all pending messages."""
        messages = []
        while not self.inbox.empty():
            try:
                messages.append(self.inbox.get_nowait())
            except queue.Empty:
                break
        return messages
    
    def broadcast_discovery(self, trajectory: Trajectory) -> SwarmMessage:
        """Create a message announcing a good discovery."""
        return SwarmMessage(
            sender_id=self.agent_id,
            content=f"Found solution with reward {trajectory.reward:.3f}",
            message_type="discovery",
            data={"reward": trajectory.reward, "temperature": trajectory.temperature},
        )


# =============================================================================
# SWARM COORDINATOR: Orchestrates the swarm
# =============================================================================

class SwarmCoordinator:
    """
    Coordinates multiple agents in the swarm.
    
    Responsibilities:
    - Spawn and manage agents
    - Distribute tasks
    - Aggregate results
    - Enable inter-agent communication
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        reward_fn: Callable[[str, str], float],
        config: Optional[SwarmConfig] = None,
    ):
        """
        Initialize the swarm coordinator.
        
        Args:
            model: Base model (will be shared or copied)
            tokenizer: Tokenizer
            reward_fn: Reward function for evaluating generations
            config: Swarm configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config or SwarmConfig()
        
        # Agent registry
        self.agents: Dict[str, SwarmAgent] = {}
        
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Results storage
        self.all_trajectories: List[Trajectory] = []
        self.best_trajectories: List[Trajectory] = []
        
        # Communication bus
        self.message_bus: List[SwarmMessage] = []
        
        # Initialize agents
        self._initialize_agents()
        
        logger.info(f"SwarmCoordinator initialized with {len(self.agents)} agents")
    
    def _initialize_agents(self) -> None:
        """Initialize the swarm agents."""
        agent_idx = 0
        
        # Create explorers
        for _ in range(self.config.num_explorers):
            agent_id = f"explorer_{agent_idx}"
            self.agents[agent_id] = SwarmAgent(
                agent_id=agent_id,
                role=SwarmRole.EXPLORER,
                model=self.model,
                tokenizer=self.tokenizer,
                reward_fn=self.reward_fn,
                config=self.config,
            )
            agent_idx += 1
        
        # Create exploiters
        for _ in range(self.config.num_exploiters):
            agent_id = f"exploiter_{agent_idx}"
            self.agents[agent_id] = SwarmAgent(
                agent_id=agent_id,
                role=SwarmRole.EXPLOITER,
                model=self.model,
                tokenizer=self.tokenizer,
                reward_fn=self.reward_fn,
                config=self.config,
            )
            agent_idx += 1
    
    def swarm_explore(
        self,
        prompts: List[str],
        samples_per_prompt: int = 1,
    ) -> List[Trajectory]:
        """
        Run parallel exploration across all agents.
        
        Args:
            prompts: List of prompts to explore
            samples_per_prompt: Samples each agent generates per prompt
            
        Returns:
            List of all trajectories, sorted by reward
        """
        all_trajectories = []
        futures: List[Future] = []
        
        # Submit tasks to all agents
        for prompt in prompts:
            for agent in self.agents.values():
                future = self.executor.submit(
                    agent.multi_explore,
                    prompt,
                    samples_per_prompt,
                )
                futures.append((future, agent.agent_id))
        
        # Collect results
        for future, agent_id in futures:
            try:
                trajectories = future.result(timeout=self.config.timeout_seconds)
                all_trajectories.extend(trajectories)
                
                if self.config.log_agent_outputs:
                    logger.debug(
                        f"Agent {agent_id} returned {len(trajectories)} trajectories"
                    )
            except Exception as e:
                logger.warning(f"Agent {agent_id} failed: {e}")
        
        # Store all trajectories
        self.all_trajectories.extend(all_trajectories)
        
        # Sort by reward
        all_trajectories.sort(key=lambda t: t.reward, reverse=True)
        
        # Apply diversity bonus
        if self.config.diversity_bonus > 0:
            all_trajectories = self._apply_diversity_bonus(all_trajectories)
        
        # Keep top-k
        self.best_trajectories = all_trajectories[:self.config.top_k_trajectories]
        
        # Broadcast discoveries
        if self.config.communication_enabled:
            self._broadcast_best_discoveries()
        
        return all_trajectories
    
    def _apply_diversity_bonus(
        self,
        trajectories: List[Trajectory],
    ) -> List[Trajectory]:
        """
        Apply diversity bonus to encourage varied solutions.
        
        Uses simple token overlap as diversity metric.
        """
        if len(trajectories) < 2:
            return trajectories
        
        # Calculate pairwise diversity
        for i, traj in enumerate(trajectories):
            diversity_scores = []
            traj_tokens = set(traj.generation.lower().split())
            
            for j, other in enumerate(trajectories):
                if i == j:
                    continue
                other_tokens = set(other.generation.lower().split())
                
                # Jaccard distance
                if traj_tokens or other_tokens:
                    intersection = len(traj_tokens & other_tokens)
                    union = len(traj_tokens | other_tokens)
                    diversity = 1 - (intersection / union) if union > 0 else 1.0
                    diversity_scores.append(diversity)
            
            if diversity_scores:
                avg_diversity = sum(diversity_scores) / len(diversity_scores)
                
                # Add bonus if diverse enough
                if avg_diversity > self.config.min_diversity_threshold:
                    bonus = self.config.diversity_bonus * avg_diversity
                    traj.metadata["diversity_bonus"] = bonus
                    traj.reward += bonus
        
        # Re-sort after bonus
        trajectories.sort(key=lambda t: t.reward, reverse=True)
        
        return trajectories
    
    def _broadcast_best_discoveries(self) -> None:
        """Broadcast best discoveries to all agents."""
        if not self.best_trajectories:
            return
        
        best = self.best_trajectories[0]
        
        for agent in self.agents.values():
            if agent.agent_id != best.agent_id:
                message = SwarmMessage(
                    sender_id="coordinator",
                    content=f"Best solution found: reward {best.reward:.3f}",
                    message_type="discovery",
                    data={
                        "reward": best.reward,
                        "agent_id": best.agent_id,
                        "role": best.role.value,
                    },
                )
                agent.receive_message(message)
        
        self.message_bus.append(message)
    
    def aggregate_trajectories(
        self,
        method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate trajectories using the specified method.
        
        Args:
            method: Aggregation method (uses config default if None)
            
        Returns:
            Aggregation result
        """
        method = method or self.config.aggregation_method
        
        if not self.best_trajectories:
            return {"error": "No trajectories to aggregate"}
        
        if method == "best":
            # Simply return the best trajectory
            best = self.best_trajectories[0]
            return {
                "method": "best",
                "trajectory": best.to_dict(),
                "reward": best.reward,
            }
        
        elif method == "weighted":
            # Weighted combination based on rewards
            total_reward = sum(t.reward for t in self.best_trajectories if t.reward > 0)
            if total_reward == 0:
                return self.aggregate_trajectories("best")
            
            weights = [t.reward / total_reward for t in self.best_trajectories if t.reward > 0]
            
            return {
                "method": "weighted",
                "trajectories": [t.to_dict() for t in self.best_trajectories],
                "weights": weights,
                "avg_reward": sum(t.reward for t in self.best_trajectories) / len(self.best_trajectories),
            }
        
        elif method == "ensemble":
            # Return all top trajectories for ensemble
            return {
                "method": "ensemble",
                "trajectories": [t.to_dict() for t in self.best_trajectories],
                "count": len(self.best_trajectories),
                "rewards": [t.reward for t in self.best_trajectories],
            }
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get swarm statistics."""
        if not self.all_trajectories:
            return {"status": "no_data"}
        
        rewards = [t.reward for t in self.all_trajectories]
        
        # Per-role statistics
        role_stats = {}
        for role in SwarmRole:
            role_trajs = [t for t in self.all_trajectories if t.role == role]
            if role_trajs:
                role_rewards = [t.reward for t in role_trajs]
                role_stats[role.value] = {
                    "count": len(role_trajs),
                    "avg_reward": sum(role_rewards) / len(role_rewards),
                    "max_reward": max(role_rewards),
                }
        
        return {
            "total_trajectories": len(self.all_trajectories),
            "avg_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "std_reward": self._std(rewards),
            "num_agents": len(self.agents),
            "role_statistics": role_stats,
            "messages_sent": len(self.message_bus),
        }
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def shutdown(self) -> None:
        """Shutdown the swarm."""
        self.executor.shutdown(wait=True)
        logger.info("SwarmCoordinator shutdown complete")


# =============================================================================
# SWARM TRAINER: High-level training interface
# =============================================================================

class SwarmTrainer:
    """
    Swarm-based training using multi-agent exploration.
    
    Combines swarm intelligence with RL training:
    1. Swarm explores solution space in parallel
    2. Best trajectories are aggregated
    3. Policy is updated using aggregated rewards
    
    Example:
        ```python
        swarm_trainer = SwarmTrainer(model, tokenizer, reward_fn)
        
        # Train with swarm exploration
        results = swarm_trainer.train(
            train_prompts=prompts,
            num_iterations=100,
        )
        ```
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        reward_fn: Callable[[str, str], float],
        config: Optional[SwarmConfig] = None,
        optimizer: Optional[Any] = None,
    ):
        """
        Initialize the swarm trainer.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer
            reward_fn: Reward function
            config: Swarm configuration
            optimizer: Optional optimizer for policy updates
        """
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config or SwarmConfig()
        self.optimizer = optimizer
        
        # Create swarm coordinator
        self.coordinator = SwarmCoordinator(
            model=model,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            config=self.config,
        )
        
        # Training state
        self.iteration = 0
        self.training_history: List[Dict[str, Any]] = []
        
        logger.info("SwarmTrainer initialized")
    
    def train_step(
        self,
        prompts: List[str],
    ) -> Dict[str, Any]:
        """
        Execute one swarm training step.
        
        Args:
            prompts: Batch of prompts
            
        Returns:
            Step results including trajectories and statistics
        """
        self.iteration += 1
        
        # Swarm exploration
        trajectories = self.coordinator.swarm_explore(
            prompts=prompts,
            samples_per_prompt=1,
        )
        
        # Aggregate
        aggregation = self.coordinator.aggregate_trajectories()
        
        # Get statistics
        stats = self.coordinator.get_statistics()
        
        # Create step result
        result = {
            "iteration": self.iteration,
            "num_trajectories": len(trajectories),
            "aggregation": aggregation,
            "statistics": stats,
        }
        
        # Store in history
        self.training_history.append(result)
        
        # Log
        logger.info(
            f"Swarm step {self.iteration}: "
            f"{len(trajectories)} trajectories, "
            f"avg reward: {stats.get('avg_reward', 0):.4f}, "
            f"max reward: {stats.get('max_reward', 0):.4f}"
        )
        
        return result
    
    def train(
        self,
        train_prompts: List[str],
        num_iterations: int = 100,
        batch_size: int = 4,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Full swarm training loop.
        
        Args:
            train_prompts: All training prompts
            num_iterations: Number of training iterations
            batch_size: Prompts per iteration
            callback: Optional callback after each step
            
        Returns:
            Training summary
        """
        logger.info(f"Starting swarm training: {num_iterations} iterations")
        
        for iteration in range(num_iterations):
            # Sample batch
            import random
            batch = random.sample(train_prompts, min(batch_size, len(train_prompts)))
            
            # Train step
            result = self.train_step(batch)
            
            # Callback
            if callback:
                callback(result)
            
            # Sync agents periodically
            if iteration % self.config.sync_every_n_steps == 0:
                self._sync_agents()
        
        # Final summary
        summary = self._create_summary()
        
        logger.info(f"Swarm training complete: {summary['total_trajectories']} trajectories")
        
        return summary
    
    def _sync_agents(self) -> None:
        """Synchronize agents (share best solutions)."""
        for agent in self.coordinator.agents.values():
            messages = agent.process_messages()
            if messages:
                logger.debug(f"Agent {agent.agent_id} processed {len(messages)} messages")
    
    def _create_summary(self) -> Dict[str, Any]:
        """Create training summary."""
        if not self.training_history:
            return {"status": "no_training_data"}
        
        all_rewards = []
        for step in self.training_history:
            stats = step.get("statistics", {})
            if "avg_reward" in stats:
                all_rewards.append(stats["avg_reward"])
        
        return {
            "total_iterations": len(self.training_history),
            "total_trajectories": sum(
                s.get("num_trajectories", 0) for s in self.training_history
            ),
            "avg_reward_over_training": sum(all_rewards) / max(len(all_rewards), 1),
            "final_statistics": self.coordinator.get_statistics(),
        }
    
    def get_best_trajectories(self) -> List[Trajectory]:
        """Get the best trajectories found during training."""
        return self.coordinator.best_trajectories
    
    def export_history(self, filepath: str) -> None:
        """Export training history to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        logger.info(f"Exported training history to: {filepath}")
    
    def shutdown(self) -> None:
        """Shutdown the swarm trainer."""
        self.coordinator.shutdown()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_swarm_trainer(
    model: Any,
    tokenizer: Any,
    reward_fn: Callable[[str, str], float],
    num_agents: int = 4,
    **kwargs,
) -> SwarmTrainer:
    """
    Factory function to create a SwarmTrainer.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        reward_fn: Reward function
        num_agents: Total number of agents
        **kwargs: Additional config parameters
        
    Returns:
        Configured SwarmTrainer
    """
    # Split agents between explorers and exploiters
    num_explorers = num_agents // 2
    num_exploiters = num_agents - num_explorers
    
    config = SwarmConfig(
        num_agents=num_agents,
        num_explorers=num_explorers,
        num_exploiters=num_exploiters,
        **kwargs,
    )
    
    return SwarmTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        config=config,
    )


def create_swarm_coordinator(
    model: Any,
    tokenizer: Any,
    reward_fn: Callable[[str, str], float],
    **kwargs,
) -> SwarmCoordinator:
    """
    Factory function to create a SwarmCoordinator.
    
    Args:
        model: Base model
        tokenizer: Tokenizer
        reward_fn: Reward function
        **kwargs: Config parameters
        
    Returns:
        Configured SwarmCoordinator
    """
    config = SwarmConfig(**kwargs)
    return SwarmCoordinator(
        model=model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        config=config,
    )

