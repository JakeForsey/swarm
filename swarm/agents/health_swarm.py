import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Callable
from dataclasses import dataclass
from swarm.env import State

@dataclass
class HealthConfig:
    """Configuration for health-based swarm behavior"""
    # Movement parameters
    attack_speed: float = 0.2  # Speed when attacking
    flee_speed: float = 0.15   # Speed when fleeing

def create_health_agent(config: HealthConfig) -> Callable:
    """Create a health-based swarm agent that attacks or flees based on relative health."""
    def act(state: State, team: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Get team positions and velocities
        if team == 1:
            ally_x, ally_y = state.x1, state.y1
            ally_health = state.health1
            enemy_x, enemy_y = state.x2, state.y2
            enemy_health = state.health2
        else:
            ally_x, ally_y = state.x2, state.y2
            ally_health = state.health2
            enemy_x, enemy_y = state.x1, state.y1
            enemy_health = state.health1

        batch_size = ally_x.shape[0]
        num_allies = ally_x.shape[1]
        
        # Calculate distances to all enemies
        # Shape: (batch_size, num_allies, num_enemies)
        dx = ally_x[:, :, None] - enemy_x[:, None, :]
        dy = ally_y[:, :, None] - enemy_y[:, None, :]
        
        # Handle wrapping
        dx = jnp.where(dx > 0.5, dx - 1.0, dx)
        dx = jnp.where(dx < -0.5, dx + 1.0, dx)
        dy = jnp.where(dy > 0.5, dy - 1.0, dy)
        dy = jnp.where(dy < -0.5, dy + 1.0, dy)
        
        # Calculate distances
        distances = jnp.sqrt(dx**2 + dy**2) + 1e-5
        
        # Create mask for alive enemies (health > 0)
        alive_enemies = enemy_health > 0  # Shape: (batch_size, num_enemies)
        
        # Set distance to dead enemies to infinity so they won't be chosen
        distances = jnp.where(alive_enemies[:, None, :], distances, jnp.inf)
        
        # Find nearest enemy for each agent
        nearest_enemy_idx = jnp.argmin(distances, axis=2)  # Shape: (batch_size, num_allies)
        
        # Get health of nearest enemies
        batch_idx = jnp.arange(batch_size)[:, None]
        agent_idx = jnp.arange(num_allies)[None, :]
        nearest_enemy_health = enemy_health[batch_idx, nearest_enemy_idx]
        
        # Compare health with nearest enemy
        health_diff = ally_health - nearest_enemy_health
        should_attack = health_diff >= 0
        # should_attack = health_diff > config.health_threshold
        
        # Get direction to nearest enemy
        nearest_enemy_dx = dx[batch_idx, agent_idx, nearest_enemy_idx]
        nearest_enemy_dy = dy[batch_idx, agent_idx, nearest_enemy_idx]
        
        # Calculate movement based on health comparison
        move_speed = jnp.where(should_attack, -config.attack_speed, config.flee_speed)
        dx = nearest_enemy_dx * move_speed
        dy = nearest_enemy_dy * move_speed
                
        return dx, dy
    
    return act

# Default configuration
DEFAULT_CONFIG = HealthConfig()

# Create default agent
default_agent = create_health_agent(DEFAULT_CONFIG)

def act(state: State, team: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Main act function that uses the default configuration."""
    return default_agent(state, team, key)
