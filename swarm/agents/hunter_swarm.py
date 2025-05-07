import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple

from swarm.env import State

@dataclass
class HunterConfig:
    # Formation parameters
    formation_scale: float = 0.2  # Tighter formation
    formation_weight: float = 0.6  # More cohesion to prevent isolation
    
    # Movement parameters
    attack_speed: float = 0.2  # Slightly slower but more controlled
    pursue_speed: float = 0.18  # Close to attack speed for better coordination
    retreat_speed: float = 0.12  # Slower retreat to stay with group
    
    # Tactical parameters
    focus_fire_radius: float = 0.15  # Tighter focus fire
    bait_threshold: float = 0.85  # Fewer bait agents
    surround_radius: float = 0.25  # Tighter surround radius
    
    # Combat parameters
    engage_threshold: float = 0.65  # More careful engagement
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

        # Find a single primary target for all agents (most damaged and central)
        enemy_health_sum = jnp.sum(enemy_health, axis=1, keepdims=True)  # Total health per enemy
        enemy_center_x = jnp.mean(enemy_x, axis=1, keepdims=True)  # Center of enemy swarm
        enemy_center_y = jnp.mean(enemy_y, axis=1, keepdims=True)
        
        # Distance from enemy center
        dx_to_center = enemy_x - enemy_center_x
        dy_to_center = enemy_y - enemy_center_y
        dist_to_center = jnp.sqrt(dx_to_center**2 + dy_to_center**2)
        
        # Score combines health and centrality
        target_scores = enemy_health + dist_to_center * 0.3
        primary_target = jnp.argmin(target_scores, axis=1, keepdims=True)
        
        # Calculate target positions
        target_x = jnp.take_along_axis(enemy_x, primary_target, axis=1)  # Shape: (batch_size, 1)
        target_y = jnp.take_along_axis(enemy_y, primary_target, axis=1)  # Shape: (batch_size, 1)
        
        dx_to_target = target_x - own_x
        dy_to_target = target_y - own_y
        dist_to_target = jnp.sqrt(dx_to_target**2 + dy_to_target**2)
        
        # Identify high-health agents to act as bait
        is_bait = own_health > config.bait_threshold
        
        # Calculate intercept positions (predict enemy movement)
        intercept_x = target_x + dx_to_target * 0.15  # Slightly reduced prediction
        intercept_y = target_y + dy_to_target * 0.15
        
        # Calculate surround positions
        num_agents = own_x.shape[1]
        angles = jnp.linspace(0, 2*jnp.pi, num_agents, endpoint=False)
        surround_x = target_x + config.surround_radius * jnp.cos(angles)[None, :]
        surround_y = target_y + config.surround_radius * jnp.sin(angles)[None, :]
        
        # Calculate move directions
        dx = jnp.where(is_bait, 
                      dx_to_target,  # Bait moves directly at target
                      surround_x - own_x)  # Others move to surround positions
        dy = jnp.where(is_bait,
                      dy_to_target,
                      surround_y - own_y)
        
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
        
        # Add formation cohesion
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
