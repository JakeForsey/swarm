import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple

from swarm.env import State

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

@jax.jit
def act( 
    t: jnp.ndarray,
    key: jnp.ndarray,
    ally_x: jnp.ndarray,
    ally_y: jnp.ndarray,
    ally_vx: jnp.ndarray,
    ally_vy: jnp.ndarray,
    ally_health: jnp.ndarray,
    enemy_y: jnp.ndarray,
    enemy_x: jnp.ndarray,
    enemy_vx: jnp.ndarray,
    enemy_vy: jnp.ndarray,
    enemy_health: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Find the most damaged enemy (primary target)
    enemy_total_health = enemy_health  # Each enemy's health
    primary_target = jnp.argmin(enemy_total_health, axis=1, keepdims=True)  # Shape: (batch_size, 1)
    
    # Calculate distances to primary target
    target_x = jnp.take_along_axis(enemy_x, primary_target, axis=1)  # Shape: (batch_size, 1)
    target_y = jnp.take_along_axis(enemy_y, primary_target, axis=1)  # Shape: (batch_size, 1)
    
    dx_to_target = target_x - ally_x  # Will broadcast correctly
    dy_to_target = target_y - ally_y  # Will broadcast correctly
    dist_to_target = jnp.sqrt(dx_to_target**2 + dy_to_target**2)
    
    # Identify high-health agents to act as bait
    is_bait = ally_health > bait_threshold
    
    # Calculate surround positions around primary target
    batch_size = ally_x.shape[0]
    num_agents = ally_x.shape[1]
    angles = jnp.linspace(0, 2*jnp.pi, num_agents, endpoint=False)
    
    # Surround position relative to target
    surround_x = surround_radius * jnp.cos(angles)  # Shape: (num_agents,)
    surround_y = surround_radius * jnp.sin(angles)  # Shape: (num_agents,)
    
    # Add target position to get absolute surround positions
    surround_x = target_x + surround_x[None, :]  # Shape: (batch_size, num_agents)
    surround_y = target_y + surround_y[None, :]  # Shape: (batch_size, num_agents)
    
    # Calculate move directions
    dx = jnp.where(is_bait, 
                    dx_to_target,  # Bait moves toward target
                    surround_x - ally_x)  # Others move to surround
    dy = jnp.where(is_bait,
                    dy_to_target,  # Bait moves toward target
                    surround_y - ally_y)  # Others move to surround
    
    # Normalize directions
    magnitude = jnp.sqrt(dx**2 + dy**2) + 1e-10
    dx = dx / magnitude
    dy = dy / magnitude
    
    # Adjust speed based on role and health
    speed = jnp.where(
        is_bait,
        jnp.where(ally_health > engage_threshold,
                    attack_speed,  # Healthy bait moves fast
                    retreat_speed),  # Damaged bait retreats
        pursue_speed  # Surrounding agents move at pursuit speed
    )
    
    # Apply speed to movement
    dx = dx * speed
    dy = dy * speed
    
    # Add formation cohesion
    center_x = jnp.mean(ally_x, axis=1, keepdims=True)
    center_y = jnp.mean(ally_y, axis=1, keepdims=True)
    dx = dx + formation_weight * (center_x - ally_x)
    dy = dy + formation_weight * (center_y - ally_y)
    
    return dx, dy
