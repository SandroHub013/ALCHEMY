"""
Example: Using the new integrations (Meta-Agent, Adaptive Trainer, Swarm Trainer)

This example demonstrates how to use the three new modules inspired by:
- PocketFlow: Meta-Agent for dynamic agent/SOP generation
- AgentFlow: Adaptive Trainer for dynamic optimization
- claude-flow: Swarm Trainer for multi-agent orchestration

Run this example with:
    python examples/integrations_example.py
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# EXAMPLE 1: Meta-Agent (PocketFlow-inspired)
# =============================================================================

def example_meta_agent():
    """
    Demonstrates the Meta-Agent: agents that generate other agents.
    
    Key features:
    - Generate specialized agent blueprints from task descriptions
    - Create custom reward functions dynamically
    - Generate SOPs on-demand
    - Spawn child agents for subtasks
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Meta-Agent (Inspired by PocketFlow)")
    print("="*70 + "\n")
    
    from agent import MetaAgent, AgentType, create_meta_agent
    
    # Create meta-agent (no model needed for basic functionality)
    meta = create_meta_agent()
    
    # --- Generate a specialized coding agent ---
    print("1. Generating a coding agent blueprint...")
    coding_blueprint = meta.generate_agent_blueprint(
        task="Write efficient Python functions with comprehensive error handling and type hints",
        name="advanced_python_coder"
    )
    
    print(f"   Name: {coding_blueprint.name}")
    print(f"   Type: {coding_blueprint.agent_type.value}")
    print(f"   Temperature: {coding_blueprint.temperature}")
    print(f"   Max tokens: {coding_blueprint.max_new_tokens}")
    print(f"   System prompt preview: {coding_blueprint.system_prompt[:100]}...")
    
    # --- Generate a reasoning agent ---
    print("\n2. Generating a reasoning agent blueprint...")
    reasoning_blueprint = meta.generate_agent_blueprint(
        task="Solve complex mathematical problems with step-by-step explanations"
    )
    
    print(f"   Name: {reasoning_blueprint.name}")
    print(f"   Type: {reasoning_blueprint.agent_type.value}")
    print(f"   Max tokens: {reasoning_blueprint.max_new_tokens} (increased for complex tasks)")
    
    # --- Generate a custom SOP ---
    print("\n3. Generating a custom SOP...")
    sop = meta.generate_sop(
        task="Review and refactor legacy Python code for performance",
        name="code_refactoring_procedure"
    )
    
    print(f"   SOP Name: {sop.name}")
    print(f"   Category: {sop.category}")
    print(f"   Steps:")
    for i, step in enumerate(sop.steps, 1):
        print(f"      {i}. {step['action']}")
    
    # --- Spawn a child agent ---
    print("\n4. Spawning a specialized child agent...")
    child = meta.spawn_child_agent(
        parent_blueprint=coding_blueprint,
        specialization="Focus on async/await patterns and concurrency"
    )
    
    print(f"   Child name: {child.name}")
    print(f"   Parent: {child.metadata.get('parent')}")
    print(f"   Specialization: {child.metadata.get('specialization')}")
    
    # --- Generate custom reward function ---
    print("\n5. Generating a custom reward function...")
    reward_fn = meta.generate_reward_function(
        task="Write well-documented code",
        blueprint=coding_blueprint
    )
    
    # Test the reward function
    test_generation = '''
```python
def calculate_sum(numbers: list[int]) -> int:
    """Calculate the sum of a list of numbers.
    
    Args:
        numbers: List of integers to sum
        
    Returns:
        The sum of all numbers
    """
    return sum(numbers)
```
'''
    reward = reward_fn("Write a sum function", test_generation)
    print(f"   Test reward: {reward:.3f}")
    
    # --- List all generated agents ---
    print("\n6. All generated agents:")
    for agent in meta.list_generated_agents():
        print(f"   - {agent['name']} ({agent['type']})")
    
    print("\n✅ Meta-Agent example complete!")
    return meta


# =============================================================================
# EXAMPLE 2: Adaptive Trainer (AgentFlow-inspired)
# =============================================================================

def example_adaptive_trainer():
    """
    Demonstrates the Adaptive Trainer: dynamic optimization during training.
    
    Key features:
    - Real-time monitoring of training metrics
    - Automatic hyperparameter adjustment
    - Curriculum learning (progressive difficulty)
    - Early stopping detection
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Adaptive Trainer (Inspired by AgentFlow)")
    print("="*70 + "\n")
    
    from agent import (
        AdaptiveTrainer, 
        AdaptiveConfig,
        TrainingState,
        create_adaptive_trainer
    )
    
    # Create adaptive trainer with curriculum learning
    config = AdaptiveConfig(
        enable_curriculum=True,
        initial_difficulty=0.3,
        patience=5,
        early_stop_patience=15,
    )
    adaptive = AdaptiveTrainer(config)
    
    print("1. Simulating training steps...")
    
    # Simulate training with decreasing loss (improving)
    simulated_training = [
        # (step, loss, reward)
        (1, 2.5, 0.1),
        (2, 2.3, 0.15),
        (3, 2.1, 0.2),
        (4, 1.9, 0.25),
        (5, 1.8, 0.3),
        (6, 1.7, 0.35),
        (7, 1.6, 0.4),
        (8, 1.55, 0.45),
        (9, 1.5, 0.5),
        (10, 1.48, 0.52),
        # Plateau starts
        (11, 1.47, 0.53),
        (12, 1.47, 0.53),
        (13, 1.46, 0.54),
        (14, 1.46, 0.54),
        (15, 1.46, 0.54),
    ]
    
    for step, loss, reward in simulated_training:
        actions = adaptive.step(
            step=step,
            loss=loss,
            reward_mean=reward,
            learning_rate=2e-4,
            temperature=0.7,
        )
        
        # Report state changes
        state = adaptive.get_state()
        if actions:
            for action in actions:
                print(f"   Step {step}: State={state.value}, Action={action.action.value}")
                print(f"            Reason: {action.reason}")
    
    # --- Show curriculum progress ---
    print(f"\n2. Curriculum Learning Progress:")
    print(f"   Current difficulty: {adaptive.get_current_difficulty():.2f}")
    
    # --- Show summary ---
    print("\n3. Training Summary:")
    summary = adaptive.get_summary()
    print(f"   Current state: {summary['current_state']}")
    print(f"   Best loss: {summary['best_loss']:.4f}")
    print(f"   Steps without improvement: {summary['steps_without_improvement']}")
    print(f"   Recommended LR: {summary['current_lr']:.2e}")
    print(f"   Total adaptations: {summary['total_adaptations']}")
    
    # --- Should we stop? ---
    print(f"\n4. Should stop training? {adaptive.should_stop()}")
    
    print("\n✅ Adaptive Trainer example complete!")
    return adaptive


# =============================================================================
# EXAMPLE 3: Swarm Trainer (claude-flow-inspired)
# =============================================================================

def example_swarm_trainer():
    """
    Demonstrates the Swarm Trainer: multi-agent orchestration.
    
    Key features:
    - Parallel exploration with multiple agents
    - Explorer/Exploiter role division
    - Trajectory aggregation
    - Inter-agent communication
    
    Note: This example uses mock objects since we don't have a loaded model.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Swarm Trainer (Inspired by claude-flow)")
    print("="*70 + "\n")
    
    from agent import (
        SwarmConfig,
        SwarmRole,
        Trajectory,
    )
    
    # Since we don't have a model loaded, we'll demonstrate the concepts
    print("1. Swarm Configuration:")
    config = SwarmConfig(
        num_agents=4,
        num_explorers=2,
        num_exploiters=2,
        explorer_temperature=0.9,
        exploiter_temperature=0.3,
        top_k_trajectories=4,
        diversity_bonus=0.1,
    )
    
    print(f"   Total agents: {config.num_agents}")
    print(f"   Explorers: {config.num_explorers} (temp={config.explorer_temperature})")
    print(f"   Exploiters: {config.num_exploiters} (temp={config.exploiter_temperature})")
    
    # --- Simulate trajectories ---
    print("\n2. Simulated Swarm Exploration:")
    
    mock_trajectories = [
        Trajectory(
            agent_id="explorer_0",
            prompt="Write a sorting function",
            generation="def sort(arr): return sorted(arr)",
            reward=0.6,
            role=SwarmRole.EXPLORER,
            temperature=0.9,
        ),
        Trajectory(
            agent_id="explorer_1",
            prompt="Write a sorting function",
            generation="def quicksort(arr): ...",  # More diverse
            reward=0.75,
            role=SwarmRole.EXPLORER,
            temperature=0.9,
        ),
        Trajectory(
            agent_id="exploiter_2",
            prompt="Write a sorting function",
            generation="def sort(arr): return sorted(arr)",
            reward=0.65,
            role=SwarmRole.EXPLOITER,
            temperature=0.3,
        ),
        Trajectory(
            agent_id="exploiter_3",
            prompt="Write a sorting function",
            generation="def bubble_sort(arr): ...",
            reward=0.55,
            role=SwarmRole.EXPLOITER,
            temperature=0.3,
        ),
    ]
    
    for traj in mock_trajectories:
        print(f"   Agent {traj.agent_id} ({traj.role.value}): reward={traj.reward:.2f}")
    
    # --- Aggregation ---
    print("\n3. Trajectory Aggregation (Best Method):")
    best = max(mock_trajectories, key=lambda t: t.reward)
    print(f"   Best trajectory: {best.agent_id}")
    print(f"   Reward: {best.reward:.2f}")
    print(f"   Role: {best.role.value}")
    
    # --- Role analysis ---
    print("\n4. Role Performance Analysis:")
    explorer_rewards = [t.reward for t in mock_trajectories if t.role == SwarmRole.EXPLORER]
    exploiter_rewards = [t.reward for t in mock_trajectories if t.role == SwarmRole.EXPLOITER]
    
    print(f"   Explorers avg reward: {sum(explorer_rewards)/len(explorer_rewards):.3f}")
    print(f"   Exploiters avg reward: {sum(exploiter_rewards)/len(exploiter_rewards):.3f}")
    
    # --- Swarm benefits ---
    print("\n5. Swarm Benefits:")
    print("   ✓ Parallel exploration covers more solution space")
    print("   ✓ Explorer/Exploiter balance optimizes discovery vs refinement")
    print("   ✓ Diversity bonus prevents convergence to local optima")
    print("   ✓ Inter-agent communication shares discoveries")
    
    print("\n✅ Swarm Trainer example complete!")


# =============================================================================
# COMBINED EXAMPLE: Using all three together
# =============================================================================

def example_combined():
    """
    Demonstrates using all three integrations together for advanced training.
    """
    print("\n" + "="*70)
    print("COMBINED EXAMPLE: All Three Integrations")
    print("="*70 + "\n")
    
    from agent import (
        MetaAgent,
        AdaptiveTrainer,
        AdaptiveConfig,
        SwarmConfig,
    )
    
    print("Workflow for Advanced Training Pipeline:")
    print()
    print("1. META-AGENT: Generate specialized agents for the task")
    print("   → Creates coding agent blueprint")
    print("   → Generates custom reward function")
    print("   → Produces task-specific SOP")
    print()
    print("2. SWARM TRAINER: Parallel exploration")
    print("   → Spawns multiple agents from blueprint")
    print("   → Each agent explores with different temperatures")
    print("   → Aggregates best trajectories")
    print()
    print("3. ADAPTIVE TRAINER: Monitor and optimize")
    print("   → Tracks training metrics in real-time")
    print("   → Adjusts LR when plateauing")
    print("   → Increases difficulty via curriculum")
    print()
    print("Result: Efficient, adaptive, multi-agent training system!")
    
    # Show configuration
    print("\n" + "-"*50)
    print("Example Configuration:")
    print("-"*50)
    
    meta = MetaAgent()
    adaptive_config = AdaptiveConfig(enable_curriculum=True)
    swarm_config = SwarmConfig(num_agents=4)
    
    print(f"""
meta_agent:
  enabled: true
  
adaptive_trainer:
  enabled: true
  enable_curriculum: true
  patience: {adaptive_config.patience}
  
swarm_trainer:
  enabled: true
  num_agents: {swarm_config.num_agents}
  num_explorers: {swarm_config.num_explorers}
  num_exploiters: {swarm_config.num_exploiters}
""")
    
    print("\n✅ Combined example complete!")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "#"*70)
    print("#" + " "*20 + "ALCHEMY INTEGRATIONS DEMO" + " "*21 + "#")
    print("#"*70)
    
    print("""
This demo showcases three new integrations inspired by:

• PocketFlow  → Meta-Agent (agents that generate agents)
• AgentFlow   → Adaptive Trainer (dynamic optimization)  
• claude-flow → Swarm Trainer (multi-agent orchestration)
""")
    
    # Run examples
    example_meta_agent()
    example_adaptive_trainer()
    example_swarm_trainer()
    example_combined()
    
    print("\n" + "#"*70)
    print("#" + " "*22 + "ALL EXAMPLES COMPLETE!" + " "*23 + "#")
    print("#"*70 + "\n")

