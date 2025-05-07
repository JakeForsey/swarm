import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple

from swarm.env import State

@dataclass
class FortressConfig:
    # Formation parameters
    outer_radius: float = 0.2  # Radius of defensive ring
    inner_radius: float = 0.08  # Radius of healing zone
    formation_weight: float = 0.8  # Strong formation keeping
    
    # Movement parameters
    patrol_speed: float = 0.12  # Speed of units on the perimeter
    retreat_speed: float = 0.15  # Speed of retreating units
    return_speed: float = 0.1   # Speed of healed units returning to position
    
    # Health parameters
    retreat_threshold: float = 0.7  # Health threshold to retreat to center
    return_threshold: float = 0.9   # Health threshold to return to perimeter
    
    # Combat parameters
    attack_range: float = 0.15  # Range at which to engage enemies
    rotation_speed: float = 0.1  # Speed of ring rotation

def create_fortress_agent(config: FortressConfig = FortressConfig()):
    """Creates a fortress swarm agent that maintains a defensive perimeter with healing."""
    
    def act(state, team: int, key) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Get own and enemy state based on team
        own_x = state.x1 if team == 1 else state.x2
        own_y = state.y1 if team == 1 else state.y2
        own_health = state.health1 if team == 1 else state.health2
        
        enemy_x = state.x2 if team == 1 else state.x1
        enemy_y = state.y2 if team == 1 else state.y1
        enemy_health = state.health2 if team == 1 else state.health1

        # Calculate center of our formation
        center_x = jnp.mean(own_x, axis=1, keepdims=True)
        center_y = jnp.mean(own_y, axis=1, keepdims=True)
        
        # Identify units that need healing
        needs_healing = own_health < config.retreat_threshold
        fully_healed = own_health > config.return_threshold
        
        # Calculate positions on the defensive ring
        num_agents = own_x.shape[1]
        # Add rotation based on time
        base_angles = jnp.linspace(0, 2*jnp.pi, num_agents, endpoint=False)
        rotation = config.rotation_speed * state.t[:, None]
        angles = base_angles[None, :] + rotation
        
        ring_x = center_x + config.outer_radius * jnp.cos(angles)
        ring_y = center_y + config.outer_radius * jnp.sin(angles)
        
        # Calculate positions in the healing zone (spiral pattern)
        heal_angles = jnp.linspace(0, 4*jnp.pi, num_agents, endpoint=False)
        heal_radius = jnp.linspace(0, config.inner_radius, num_agents, endpoint=False)
        heal_x = center_x + heal_radius[None, :] * jnp.cos(heal_angles)[None, :]
        heal_y = center_y + heal_radius[None, :] * jnp.sin(heal_angles)[None, :]
        
        # Calculate distances to nearest enemy
        dx_to_enemies = enemy_x[:, :, None] - own_x[:, None, :]
        dy_to_enemies = enemy_y[:, :, None] - own_y[:, None, :]
        dist_to_enemies = jnp.sqrt(dx_to_enemies**2 + dy_to_enemies**2)
        nearest_enemy_dist = jnp.min(dist_to_enemies, axis=1)
        
        # Determine if enemies are in range
        enemies_in_range = nearest_enemy_dist < config.attack_range
        
        # Calculate target positions
        # If healing: go to healing zone
        # If healed but enemies nearby: stay in position
        # If healed and safe: return to ring
        target_x = jnp.where(needs_healing, heal_x,
                           jnp.where(enemies_in_range, own_x, ring_x))
        target_y = jnp.where(needs_healing, heal_y,
                           jnp.where(enemies_in_range, own_y, ring_y))
        
        # Calculate move directions
        dx = target_x - own_x
        dy = target_y - own_y
        
        # Normalize directions
        magnitude = jnp.sqrt(dx**2 + dy**2) + 1e-10
        dx = dx / magnitude
        dy = dy / magnitude
        
        # Adjust speed based on state
        speed = jnp.where(needs_healing, config.retreat_speed,
                         jnp.where(fully_healed & ~enemies_in_range, 
                                 config.return_speed,
                                 config.patrol_speed))
        
        # Apply speed to movement
        dx = dx * speed
        dy = dy * speed
        
        # Add formation cohesion
        dx = dx + config.formation_weight * (center_x - own_x)
        dy = dy + config.formation_weight * (center_y - own_y)
        
        return dx, dy
    
    return act

# Default configuration
DEFAULT_CONFIG = FortressConfig()

# Create default agent
default_agent = create_fortress_agent(DEFAULT_CONFIG)

def act(state: State, team: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Main act function that uses the default configuration."""
    return default_agent(state, team, key) 