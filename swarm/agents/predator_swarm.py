import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple

from swarm.env import State

@dataclass
class PredatorConfig:
    # Formation parameters
    formation_scale: float = 0.3  # Scale of the formation
    formation_weight: float = 0.8  # How strongly to maintain formation
    
    # Movement parameters
    attack_speed: float = 0.2
    pursue_speed: float = 0.15
    retreat_speed: float = 0.1
    
    # Tactical parameters
    focus_fire_radius: float = 0.2  # Radius for focusing on same target
    bait_threshold: float = 0.9  # Health threshold for agents that act as bait
    surround_radius: float = 0.4  # Radius for surrounding enemies
    
    # Combat parameters
    engage_threshold: float = 0.7  # Health threshold for engaging
    disengage_threshold: float = 0.5  # Health threshold for retreating

def create_predator_agent(config: PredatorConfig = PredatorConfig()):
    """Creates a predator swarm agent specifically designed to counter health-based strategies."""
    
    def act(state, team: int, key) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Get own and enemy state based on team
        own_x = state.x1 if team == 1 else state.x2
        own_y = state.y1 if team == 1 else state.y2
        own_health = state.health1 if team == 1 else state.health2
        
        enemy_x = state.x2 if team == 1 else state.x1
        enemy_y = state.y2 if team == 1 else state.y1
        enemy_health = state.health2 if team == 1 else state.health1

        # Find the most damaged enemy (primary target)
        enemy_total_health = enemy_health  # Each enemy's health
        primary_target = jnp.argmin(enemy_total_health, axis=1, keepdims=True)  # Shape: (batch_size, 1)
        
        # Calculate distances to primary target
        target_x = jnp.take_along_axis(enemy_x, primary_target, axis=1)  # Shape: (batch_size, 1)
        target_y = jnp.take_along_axis(enemy_y, primary_target, axis=1)  # Shape: (batch_size, 1)
        
        dx_to_target = target_x - own_x  # Will broadcast correctly
        dy_to_target = target_y - own_y  # Will broadcast correctly
        dist_to_target = jnp.sqrt(dx_to_target**2 + dy_to_target**2)
        
        # Identify high-health agents to act as bait
        is_bait = own_health > config.bait_threshold
        
        # Calculate surround positions around primary target
        batch_size = own_x.shape[0]
        num_agents = own_x.shape[1]
        angles = jnp.linspace(0, 2*jnp.pi, num_agents, endpoint=False)
        
        # Surround position relative to target
        surround_x = config.surround_radius * jnp.cos(angles)  # Shape: (num_agents,)
        surround_y = config.surround_radius * jnp.sin(angles)  # Shape: (num_agents,)
        
        # Add target position to get absolute surround positions
        surround_x = target_x + surround_x[None, :]  # Shape: (batch_size, num_agents)
        surround_y = target_y + surround_y[None, :]  # Shape: (batch_size, num_agents)
        
        # Calculate move directions
        dx = jnp.where(is_bait, 
                      dx_to_target,  # Bait moves toward target
                      surround_x - own_x)  # Others move to surround
        dy = jnp.where(is_bait,
                      dy_to_target,  # Bait moves toward target
                      surround_y - own_y)  # Others move to surround
        
        # Normalize directions
        magnitude = jnp.sqrt(dx**2 + dy**2) + 1e-10
        dx = dx / magnitude
        dy = dy / magnitude
        
        # Adjust speed based on role and health
        speed = jnp.where(
            is_bait,
            jnp.where(own_health > config.engage_threshold,
                     config.attack_speed,  # Healthy bait moves fast
                     config.retreat_speed),  # Damaged bait retreats
            config.pursue_speed  # Surrounding agents move at pursuit speed
        )
        
        # Apply speed to movement
        dx = dx * speed
        dy = dy * speed
        
        # Add formation cohesion
        center_x = jnp.mean(own_x, axis=1, keepdims=True)
        center_y = jnp.mean(own_y, axis=1, keepdims=True)
        dx = dx + config.formation_weight * (center_x - own_x)
        dy = dy + config.formation_weight * (center_y - own_y)
        
        return dx, dy
    
    return act

# Default configuration
DEFAULT_CONFIG = PredatorConfig()

# Create default agent
default_agent = create_predator_agent(DEFAULT_CONFIG)

def act(state: State, team: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Main act function that uses the default configuration."""
    return default_agent(state, team, key)
