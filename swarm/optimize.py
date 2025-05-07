import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Tuple
import json
from datetime import datetime
import os

from swarm.env import State
from swarm.agents.config_swarm import (
    SwarmConfig,
    FormationConfig,
    CombatConfig,
    MovementConfig,
    create_swarm_agent
)

# Parameter ranges for optimization
PARAM_RANGES = {
    # Formation parameters
    'formation_scale': (0.3, 0.9),  # How tightly agents cluster
    'formation_shape': (0.0, 1.0),  # Shape of formation (0 = circle, 1 = line)
    'formation_weight': (0.5, 1.5), # Formation priority
    
    # Combat parameters
    'combat_aggressiveness': (0.3, 0.9),   # How aggressively to approach enemies
    'combat_attack_threshold': (0.2, 0.4), # Health threshold to start attacking
    'combat_retreat_threshold': (0.1, 0.3),# Health threshold to start retreating
    'combat_weight': (0.5, 1.5),          # Combat priority
    
    # Movement parameters
    'movement_max_speed': (0.005, 0.015), # Maximum movement speed
    'movement_smoothness': (0.6, 0.95),   # How smoothly to change direction
    'movement_damping': (0.05, 0.2)       # Velocity damping factor
}

def generate_random_params() -> Dict[str, float]:
    """Generate random parameters within the specified ranges."""
    return {
        param: np.random.uniform(low, high)
        for param, (low, high) in PARAM_RANGES.items()
    }

def create_config_from_params(params: Dict[str, float]) -> SwarmConfig:
    """Create a SwarmConfig from the parameter dictionary."""
    return SwarmConfig(
        formation=FormationConfig(
            scale=params['formation_scale'],
            shape=params['formation_shape'],
            weight=params['formation_weight']
        ),
        combat=CombatConfig(
            aggressiveness=params['combat_aggressiveness'],
            attack_threshold=params['combat_attack_threshold'],
            retreat_threshold=params['combat_retreat_threshold'],
            weight=params['combat_weight']
        ),
        movement=MovementConfig(
            max_speed=params['movement_max_speed'],
            smoothness=params['movement_smoothness'],
            damping=params['movement_damping']
        )
    )

def evaluate_agent(agent: callable, num_episodes: int = 10) -> float:
    """Evaluate an agent by running multiple episodes against multiple opponents."""
    from swarm.env import SwarmEnv
    from swarm.agents.vortex_swarm import act as vortex_act
    from swarm.agents.param_swarm import act as param_act
    from swarm.agents.squad_swarm import act as squad_act
    from swarm.agents.predator_boid import act as predator_act
    
    # List of opponent agents to test against
    opponents = [
        vortex_act,
        param_act,
        squad_act,
        predator_act
    ]
    
    total_reward = 0.0
    
    # Test against each opponent
    for opponent in opponents:
        env = SwarmEnv(episode_length=128)
        opponent_reward = 0.0
        
        for _ in range(num_episodes):
            state = env.reset()
            episode_reward = 0.0
            for step in range(env.episode_length):
                # Get actions from both agents
                key = jax.random.PRNGKey(0)
                x_action1, y_action1 = agent(state, 1, key)
                x_action2, y_action2 = opponent(state, 2, key)
                
                # Step the environment
                state, reward = env.step(state, x_action1, y_action1, x_action2, y_action2)
                if step == env.episode_length - 1:
                    episode_reward += reward.sum()
            
            opponent_reward += episode_reward
        
        # Average reward against this opponent
        total_reward += opponent_reward / num_episodes
    
    # Return average reward across all opponents
    return total_reward / len(opponents)


def optimize_agents(
    num_agents: int = 10,
    num_episodes: int = 10,
    save_best: bool = True
) -> List[Dict]:
    """Generate and evaluate multiple agents."""
    results = []
    best_reward = float('-inf')
    best_config = None
    
    for i in range(num_agents):
        print(f"\nGenerating and evaluating agent {i+1}/{num_agents}")
        
        # Generate random parameters and create config
        params = generate_random_params()
        config = create_config_from_params(params)
        
        # Create and evaluate agent
        agent = create_swarm_agent(config)
        reward = evaluate_agent(agent, num_episodes)
        
        # Store results
        result = {
            'agent_id': i,
            'params': params,
            'config': {
                'formation': {
                    'scale': config.formation.scale,
                    'shape': config.formation.shape,
                    'weight': config.formation.weight
                },
                'combat': {
                    'aggressiveness': config.combat.aggressiveness,
                    'attack_threshold': config.combat.attack_threshold,
                    'retreat_threshold': config.combat.retreat_threshold,
                    'weight': config.combat.weight
                },
                'movement': {
                    'max_speed': config.movement.max_speed,
                    'smoothness': config.movement.smoothness,
                    'damping': config.movement.damping
                }
            },
            'reward': float(reward),
            'timestamp': datetime.now().isoformat()
        }
        results.append(result)
        
        # Update best agent
        if reward > best_reward:
            best_reward = reward
            best_config = config
        
        print(f"Agent {i+1} achieved reward: {reward:.3f}")
        print(f"Parameters: {json.dumps(params, indent=2)}")
    
    # Save best configuration
    if save_best and best_config is not None:
        save_best_config(best_config, best_reward)
    
    return results

def save_best_config(config: SwarmConfig, reward: float):
    """Save the best configuration to a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"best_config_{timestamp}.py"
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)
    
    # Create a new agent function with the best configuration
    with open(filepath, 'w') as f:
        f.write(f"""from swarm.agents.config_swarm import SwarmConfig, FormationConfig, CombatConfig, MovementConfig, create_swarm_agent

# Best configuration found during optimization
# Reward: {reward:.3f}
config = SwarmConfig(
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

# Create the agent
agent = create_swarm_agent(config)
""")
    
    print(f"\nBest configuration saved to {filepath}")

def save_results(results: List[Dict], filename: str = None):
    """Save optimization results to a file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent_optimization_{timestamp}.json"
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {filepath}")

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate and evaluate agents
    results = optimize_agents(num_agents=10, num_episodes=10)
    
    # Sort results by reward
    results.sort(key=lambda x: x['reward'], reverse=True)
    
    # Save results
    save_results(results)
    
    # Print summary
    print("\nOptimization Summary:")
    print(f"Best agent reward: {results[0]['reward']:.3f}")
    print(f"Average reward: {sum(r['reward'] for r in results) / len(results):.3f}")
    print(f"Worst agent reward: {results[-1]['reward']:.3f}")
    
    # Print best configuration
    print("\nBest Configuration:")
    print(json.dumps(results[0]['config'], indent=2))

if __name__ == "__main__":
    main()
