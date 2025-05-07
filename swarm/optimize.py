import json
import importlib
import pkgutil
import os
from typing import Dict

from bayes_opt import BayesianOptimization
import jax
import jax.numpy as jnp

from swarm.env import SwarmEnv
from swarm.batch import batch_act, compute_agent_schedules
from swarm import agents as agents_module
from swarm.agents.config_swarm import SwarmConfig, FormationConfig, CombatConfig, MovementConfig, create_swarm_agent

def create_config_from_params(params: Dict[str, float]) -> SwarmConfig:
    """Create a SwarmConfig from the parameter dictionary."""
    return SwarmConfig(
        formation=FormationConfig(
            scale=float(params['formation_scale']),
            shape=float(params['formation_shape']),
            weight=float(params['formation_weight'])
        ),
        combat=CombatConfig(
            aggressiveness=float(params['aggressiveness']),
            attack_threshold=float(params['attack_threshold']),
            retreat_threshold=float(params['retreat_threshold']),
            weight=float(params['combat_weight'])
        ),
        movement=MovementConfig(
            max_speed=float(params['max_speed']),
            smoothness=float(params['smoothness']),
            damping=float(params['damping'])
        )
    )

def params_to_array(params: Dict[str, float]) -> jnp.ndarray:
    """Convert parameter dictionary to array."""
    return jnp.array([
        params['formation_scale'],
        params['formation_shape'],
        params['formation_weight'],
        params['aggressiveness'],
        params['attack_threshold'],
        params['retreat_threshold'],
        params['combat_weight'],
        params['max_speed'],
        params['smoothness'],
        params['damping']
    ])


def evaluate_params(**params: Dict[str, float]) -> float:
    """Evaluate parameters by running an episode."""
    # Create config and agent
    config = SwarmConfig(
        formation=FormationConfig(
            scale=jnp.asarray(params['formation_scale'], dtype=jnp.float32),
            shape=jnp.asarray(params['formation_shape'], dtype=jnp.float32),
            weight=jnp.asarray(params['formation_weight'], dtype=jnp.float32)
        ),
        combat=CombatConfig(
            aggressiveness=jnp.asarray(params['aggressiveness'], dtype=jnp.float32),
            attack_threshold=jnp.asarray(params['attack_threshold'], dtype=jnp.float32),
            retreat_threshold=jnp.asarray(params['retreat_threshold'], dtype=jnp.float32),
            weight=jnp.asarray(params['combat_weight'], dtype=jnp.float32)
        ),
        movement=MovementConfig(
            max_speed=jnp.asarray(params['max_speed'], dtype=jnp.float32),
            smoothness=jnp.asarray(params['smoothness'], dtype=jnp.float32),
            damping=jnp.asarray(params['damping'], dtype=jnp.float32)
        )
    )

    class Agent:
        def __init__(self):
            self.act = create_swarm_agent(config)
    agent = Agent()

    # Load opponent agents
    opponents = [
        importlib.import_module(f"swarm.agents.{info.name}")
        for info in pkgutil.iter_modules(agents_module.__path__)
    ]
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

def optimize_swarm(num_iterations: int = 100) -> SwarmConfig:
    """Optimize swarm parameters using optax."""
    # Define parameter bounds
    param_bounds = {
        'formation_scale': (0.1, 0.9),
        'formation_shape': (0.1, 0.9),
        'formation_weight': (0.2, 1.4),
        'aggressiveness': (0.4, 1.2),
        'attack_threshold': (0.01, 0.6),
        'retreat_threshold': (0.01, 0.6),
        'combat_weight': (0.4, 1.4),
        'max_speed': (0.004, 0.02),
        'smoothness': (0.5, 0.98),
        'damping': (0.04, 0.18)
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

def save_best_config(config: SwarmConfig, reward: float, timestamp: str):
    """Save the best configuration to a file."""
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Save config as Python file
    config_path = f"results/best_config_{timestamp}.py"
    with open(config_path, "w") as f:
        f.write(f"""# Best configuration from optimization
from swarm.agents.config_swarm import SwarmConfig, FormationConfig, CombatConfig, MovementConfig, create_swarm_agent

BEST_CONFIG = SwarmConfig(
    formation=FormationConfig(
        scale={config.formation.scale},
        shape={config.formation.shape},
        weight={config.formation.weight}
    ),
    combat=CombatConfig(
        aggressiveness={config.combat.aggressiveness},
        attack_threshold={config.combat.attack_threshold},
        retreat_threshold={config.combat.retreat_threshold},
        weight={config.combat.weight}
    ),
    movement=MovementConfig(
        max_speed={config.movement.max_speed},
        smoothness={config.movement.smoothness},
        damping={config.movement.damping}
    )
)

# Create the best agent
best_agent = create_swarm_agent(BEST_CONFIG)

def act(state, team, key):
    return best_agent(state, team, key)
""")
    
    # Save optimization results as JSON
    results_path = f"results/agent_optimization_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": {
                "formation": {
                    "scale": config.formation.scale,
                    "shape": config.formation.shape,
                    "weight": config.formation.weight
                },
                "combat": {
                    "aggressiveness": config.combat.aggressiveness,
                    "attack_threshold": config.combat.attack_threshold,
                    "retreat_threshold": config.combat.retreat_threshold,
                    "weight": config.combat.weight
                },
                "movement": {
                    "max_speed": config.movement.max_speed,
                    "smoothness": config.movement.smoothness,
                    "damping": config.movement.damping
                }
            },
            "reward": reward,
            "timestamp": timestamp
        }, f, indent=2)

if __name__ == "__main__":
    best_config = optimize_swarm(num_iterations=100)
    print("\nBest configuration:")
    print(f"Formation scale: {best_config.formation.scale:.4f}")
    print(f"Formation shape: {best_config.formation.shape:.4f}")
    print(f"Formation weight: {best_config.formation.weight:.4f}")
    print(f"Aggressiveness: {best_config.combat.aggressiveness:.4f}")
    print(f"Attack threshold: {best_config.combat.attack_threshold:.4f}")
    print(f"Retreat threshold: {best_config.combat.retreat_threshold:.4f}")
    print(f"Combat weight: {best_config.combat.weight:.4f}")
    print(f"Max speed: {best_config.movement.max_speed:.4f}")
    print(f"Smoothness: {best_config.movement.smoothness:.4f}")
    print(f"Damping: {best_config.movement.damping:.4f}")
