import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple

from swarm.env import State

@dataclass
class HunterConfig:
    # Formation parameters
    formation_scale: float = 0.2  # Tighter formation
    formation_weight: float = 0.4  # Less cohesion for more aggressive moves
    
    # Movement parameters
    attack_speed: float = 0.25  # Faster attack speed
    pursue_speed: float = 0.2   # Faster pursuit
    retreat_speed: float = 0.15  # Faster retreat
    
    # Tactical parameters
    focus_fire_radius: float = 0.15  # Tighter focus fire
    bait_threshold: float = 0.8  # More agents can be bait
    surround_radius: float = 0.25  # Tighter surround radius
    
    # Combat parameters
    engage_threshold: float = 0.6  # More aggressive engagement
    disengage_threshold: float = 0.4  # Stay in fight longer

def create_hunter_agent(config: HunterConfig = HunterConfig()):
    """Creates a hunter swarm agent that focuses on aggressive encirclement."""
    
    def act(state, team: int, key) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Get own and enemy state based on team
        own_x = state.x1 if team == 1 else state.x2
        own_y = state.y1 if team == 1 else state.y2
        own_health = state.health1 if team == 1 else state.health2
        
        enemy_x = state.x2 if team == 1 else state.x1
        enemy_y = state.y2 if team == 1 else state.y1
        enemy_health = state.health2 if team == 1 else state.health1

        # Find closest damaged enemy as target (combines damage and proximity)
        enemy_health_matrix = enemy_health[:, :, None]  # Shape: (batch_size, num_enemies, 1)
        dx_matrix = enemy_x[:, :, None] - own_x[:, None, :]  # Shape: (batch_size, num_enemies, num_agents)
        dy_matrix = enemy_y[:, :, None] - own_y[:, None, :]
        dist_matrix = jnp.sqrt(dx_matrix**2 + dy_matrix**2)
        
        # Score combines health and distance (prefer damaged and close enemies)
        target_scores = enemy_health_matrix + dist_matrix * 0.5
        primary_target = jnp.argmin(target_scores, axis=1)  # Shape: (batch_size, num_agents)
        
        # Calculate target positions for each agent
        target_x = jnp.take_along_axis(enemy_x[:, :, None], primary_target[:, None, :], axis=1)[:, 0, :]
        target_y = jnp.take_along_axis(enemy_y[:, :, None], primary_target[:, None, :], axis=1)[:, 0, :]
        
        dx_to_target = target_x - own_x
        dy_to_target = target_y - own_y
        dist_to_target = jnp.sqrt(dx_to_target**2 + dy_to_target**2)
        
        # Identify high-health agents to act as bait
        is_bait = own_health > config.bait_threshold
        
        # Calculate intercept positions (predict enemy movement)
        intercept_x = target_x + dx_to_target * 0.2  # Predict enemy movement
        intercept_y = target_y + dy_to_target * 0.2
        
        # Calculate move directions
        dx = jnp.where(is_bait, 
                      dx_to_target,  # Bait moves directly at target
                      intercept_x - own_x)  # Others move to intercept
        dy = jnp.where(is_bait,
                      dy_to_target,
                      intercept_y - own_y)
        
        # Normalize directions
        magnitude = jnp.sqrt(dx**2 + dy**2) + 1e-10
        dx = dx / magnitude
        dy = dy / magnitude
        
        # Adjust speed based on role and health
        speed = jnp.where(
            is_bait,
            jnp.where(own_health > config.engage_threshold,
                     config.attack_speed,
                     config.retreat_speed),
            config.pursue_speed
        )
        
        # Apply speed to movement
        dx = dx * speed
        dy = dy * speed
        
        # Add lighter formation cohesion
        center_x = jnp.mean(own_x, axis=1, keepdims=True)
        center_y = jnp.mean(own_y, axis=1, keepdims=True)
        dx = dx + config.formation_weight * (center_x - own_x)
        dy = dy + config.formation_weight * (center_y - own_y)
        
        return dx, dy
    
    return act

# Default configuration
DEFAULT_CONFIG = HunterConfig()

# Create default agent
default_agent = create_hunter_agent(DEFAULT_CONFIG)

def act(state: State, team: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Main act function that uses the default configuration."""
    return default_agent(state, team, key) 