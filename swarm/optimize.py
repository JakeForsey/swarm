import json
import os
from typing import Dict

from bayes_opt import BayesianOptimization
import jax
import jax.numpy as jnp

from swarm.env import SwarmEnv
from swarm.batch import batch_act, compute_agent_schedules
from swarm.agents import load_agents
from swarm.agents.spiral_swarm import SpiralConfig, create_spiral_agent

def create_config_from_params(params: Dict[str, float]) -> SpiralConfig:
    """Create a SpiralConfig from the parameter dictionary."""
    return SpiralConfig(
        base_radius=float(params['base_radius']),
        rotation_speed=float(params['rotation_speed']),
        spiral_tightness=float(params['spiral_tightness']),
        formation_weight=float(params['formation_weight']),
        velocity_weight=float(params['velocity_weight']),
        chase_radius=float(params['chase_radius']),
        chase_weight=float(params['chase_weight']),
        min_group_size=1,
        health_aggression_scale=float(params['health_aggression_scale']),
        perception_radius=float(params['perception_radius']),
        damping=float(params['damping']),
        approach_speed=float(params['approach_speed'])
    )

def evaluate_params(**params: Dict[str, float]) -> float:
    """Evaluate parameters by running an episode."""
    # Create config and agent
    config = SpiralConfig(
        base_radius=jnp.asarray(params['base_radius'], dtype=jnp.float32),
        rotation_speed=jnp.asarray(params['rotation_speed'], dtype=jnp.float32),
        spiral_tightness=jnp.asarray(params['spiral_tightness'], dtype=jnp.float32),
        formation_weight=jnp.asarray(params['formation_weight'], dtype=jnp.float32),
        velocity_weight=jnp.asarray(params['velocity_weight'], dtype=jnp.float32),
        chase_radius=jnp.asarray(params['chase_radius'], dtype=jnp.float32),
        chase_weight=jnp.asarray(params['chase_weight'], dtype=jnp.float32),
        min_group_size=1,
        health_aggression_scale=jnp.asarray(params['health_aggression_scale'], dtype=jnp.float32),
        perception_radius=jnp.asarray(params['perception_radius'], dtype=jnp.float32),
        damping=jnp.asarray(params['damping'], dtype=jnp.float32),
        approach_speed=jnp.asarray(params['approach_speed'], dtype=jnp.float32)
    )

    class Agent:
        def __init__(self):
            self.act = create_spiral_agent(config)
    agent = Agent()

    # Load opponent agents
    opponents = load_agents()
    agents = [agent] + opponents
    num_agents = len(agents)
        
    # Compute agent schedules
    agent_schedules1 = compute_agent_schedules(num_agents, 512, 1)
    agent_schedules2 = compute_agent_schedules(num_agents, 512, 2)
    batch_size = agent_schedules1.shape[1]
    # Create environment
    env = SwarmEnv(batch_size=batch_size)
    state = env.reset()

    keys1 = jax.random.split(jax.random.PRNGKey(0), env.episode_length)
    keys2 = jax.random.split(jax.random.PRNGKey(1), env.episode_length)
    for step, (key1, key2) in enumerate(zip(keys1, keys2)):
        # Get actions from both agents
        x_action1, y_action1 = batch_act(state, agents, agent_schedules1, 1, key1)
        x_action2, y_action2 = batch_act(state, agents, agent_schedules2, 2, key2)
        
        # Step the environment
        state, reward = env.step(state, x_action1, y_action1, x_action2, y_action2)
        
        # Compute reward
        if step == env.episode_length - 1:
            team1_indices = agent_schedules1[0]
            team2_indices = agent_schedules2[0]
            final_reward = 0
            if team1_indices.any():
                final_reward += reward[team1_indices].mean()
            if team2_indices.any():
                final_reward += -1 * reward[team2_indices].mean()
    
    return final_reward

def optimize_swarm() -> SpiralConfig:
    """Optimize swarm parameters using optax."""
    # Define parameter bounds
    param_bounds = {
        'base_radius': (0.05, 0.2),
        'rotation_speed': (0.1, 0.5),
        'spiral_tightness': (0.1, 0.4),
        'formation_weight': (0.05, 0.15),
        'velocity_weight': (0.05, 0.15),
        'chase_radius': (0.3, 0.5),
        'chase_weight': (0.005, 0.02),
        'health_aggression_scale': (0.5, 1.5),
        'perception_radius': (0.2, 0.4),
        'damping': (0.05, 0.15),
        'approach_speed': (0.05, 0.2)
    }
    
    optimizer = BayesianOptimization(
        f=evaluate_params,
        pbounds=param_bounds,
        random_state=2,
    )

    optimizer.maximize(
        init_points=2,
        n_iter=10,
    )
    with open("tmp.jsonl", "w") as f:
        for iter in optimizer.res:
            f.write(json.dumps(iter) + "\n")

    print(optimizer.max)

    # Convert best parameters to config
    best_params = optimizer.max["params"]
    best_config = create_config_from_params(best_params)

    from datetime import datetime
    save_best_config(best_config, optimizer.max["target"], datetime.now().strftime("%Y%m%d_%H%M%S"))

    return best_config

def save_best_config(config: SpiralConfig, reward: float, timestamp: str):
    """Save the best configuration to a file."""
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Save config as Python file
    config_path = f"results/best_config_{timestamp}.py"
    with open(config_path, "w") as f:
        f.write(f"""# Best configuration from optimization
from swarm.agents.spiral_swarm import SpiralConfig

BEST_CONFIG = SpiralConfig(
    base_radius={config.base_radius},
    rotation_speed={config.rotation_speed},
    spiral_tightness={config.spiral_tightness},
    formation_weight={config.formation_weight},
    velocity_weight={config.velocity_weight},
    chase_radius={config.chase_radius},
    chase_weight={config.chase_weight},
    min_group_size={config.min_group_size},
    health_aggression_scale={config.health_aggression_scale},
    perception_radius={config.perception_radius},
    damping={config.damping},
    approach_speed={config.approach_speed}
)

# Create the best agent
best_agent = create_spiral_agent(BEST_CONFIG)

def act(state, team, key):
    return best_agent(state, team, key)
""")
    
    # Save optimization results as JSON
    results_path = f"results/agent_optimization_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": config.__dict__,
            "reward": reward,
            "timestamp": timestamp
        }, f, indent=2)

if __name__ == "__main__":
    best_config = optimize_swarm()
    print("\nBest configuration:")
    for field, value in best_config.__dict__.items():
        print(f"{field}: {value:.4f}")
