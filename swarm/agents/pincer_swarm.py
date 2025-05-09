import jax
import jax.numpy as jnp
from typing import Tuple, Callable

# Formation parameters
base_radius: float = 0.15  # Radius of each semi-circle
rotation_speed: float = 0.25  # Speed of rotation
pincer_angle: float = 0.3  # Angle between semi-circles (0 = closed, 1 = open)
formation_weight: float = 1.2  # Formation priority
velocity_weight: float = 0.1  # Velocity matching

# Combat parameters
chase_radius: float = 0.3  # Engagement range
chase_weight: float = 0.15  # Chase priority
min_group_size: int = 2  # Group size for engagement
health_aggression_scale: float = 1.3  # Health-based aggression
perception_radius: float = 0.35  # Enemy detection range
retreat_speed: float = 0.1  # Speed when retreating

# Movement parameters
damping: float = 0.08  # Movement damping
approach_speed: float = 0.16  # Approach speed

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
    batch_size, num_agents = ally_x.shape
    
    # Calculate centers
    ally_com_x = jnp.mean(ally_x, axis=1, keepdims=True)
    ally_com_y = jnp.mean(ally_y, axis=1, keepdims=True)
    enemy_com_x = jnp.mean(enemy_x, axis=1, keepdims=True)
    enemy_com_y = jnp.mean(enemy_y, axis=1, keepdims=True)
    
    # Calculate direction to enemy
    dx = enemy_com_x - ally_com_x
    dy = enemy_com_y - ally_com_y
    
    # Handle wrapping
    dx = jnp.where(dx > 0.5, dx - 1.0, dx)
    dx = jnp.where(dx < -0.5, dx + 1.0, dx)
    dy = jnp.where(dy > 0.5, dy - 1.0, dy)
    dy = jnp.where(dy < -0.5, dy + 1.0, dy)
    
    # Normalize direction
    dist = jnp.sqrt(dx**2 + dy**2) + 1e-5
    dx = dx / dist
    dy = dy / dist
    
    # Calculate perpendicular direction
    perp_x = -dy
    perp_y = dx
    
    # Calculate base angle for pincer formation
    base_angle = jnp.arctan2(dy, dx)  # Shape: (batch_size, 1)
    
    # Calculate time-based rotation
    rotation = t * rotation_speed  # Shape: (batch_size,)
    rotation = jnp.reshape(rotation, (batch_size, 1))  # Reshape to (batch_size, 1)
    
    # Split agents into two semi-circles
    half_agents = num_agents // 2
    angles1 = jnp.linspace(-jnp.pi/2, jnp.pi/2, half_agents)  # Shape: (half_agents,)
    angles2 = jnp.linspace(jnp.pi/2, 3*jnp.pi/2, num_agents - half_agents)  # Shape: (num_agents - half_agents,)
    
    # Reshape angles for broadcasting
    angles1 = jnp.reshape(angles1, (1, half_agents))  # Shape: (1, half_agents)
    angles2 = jnp.reshape(angles2, (1, num_agents - half_agents))  # Shape: (1, num_agents - half_agents)
    
    # Add rotation and pincer angle with proper broadcasting
    angles1 = angles1 + rotation + base_angle  # All shapes: (batch_size, 1)
    angles2 = angles2 + rotation + base_angle + jnp.pi * pincer_angle
    
    # Calculate target positions for both semi-circles
    target_x1 = ally_com_x + base_radius * jnp.cos(angles1)
    target_y1 = ally_com_y + base_radius * jnp.sin(angles1)
    target_x2 = ally_com_x + base_radius * jnp.cos(angles2)
    target_y2 = ally_com_y + base_radius * jnp.sin(angles2)
    
    # Combine target positions
    target_x = jnp.concatenate([target_x1, target_x2], axis=1)
    target_y = jnp.concatenate([target_y1, target_y2], axis=1)
    
    # Calculate formation forces
    formation_dx = (target_x - ally_x) * formation_weight
    formation_dy = (target_y - ally_y) * formation_weight
    
    # Calculate target velocities (tangential to the formation)
    target_vx1 = -base_radius * jnp.sin(angles1) * rotation_speed
    target_vy1 = base_radius * jnp.cos(angles1) * rotation_speed
    target_vx2 = -base_radius * jnp.sin(angles2) * rotation_speed
    target_vy2 = base_radius * jnp.cos(angles2) * rotation_speed
    
    # Combine target velocities
    target_vx = jnp.concatenate([target_vx1, target_vx2], axis=1)
    target_vy = jnp.concatenate([target_vy1, target_vy2], axis=1)
    
    # Calculate velocity matching forces
    velocity_match_x = (target_vx - ally_vx) * velocity_weight
    velocity_match_y = (target_vy - ally_vy) * velocity_weight
    
    # Calculate combat forces
    enemy_dx = ally_x[:, None, :] - enemy_x[:, :, None]
    enemy_dy = ally_y[:, None, :] - enemy_y[:, :, None]
    
    # Handle wrapping for combat
    enemy_dx = jnp.where(enemy_dx > 0.5, enemy_dx - 1.0, enemy_dx)
    enemy_dx = jnp.where(enemy_dx < -0.5, enemy_dx + 1.0, enemy_dx)
    enemy_dy = jnp.where(enemy_dy > 0.5, enemy_dy - 1.0, enemy_dy)
    enemy_dy = jnp.where(enemy_dy < -0.5, enemy_dy + 1.0, enemy_dy)
    
    enemy_dist = jnp.sqrt(enemy_dx**2 + enemy_dy**2)
    
    # Count nearby enemies
    nearby_enemies = jnp.sum(enemy_dist < chase_radius, axis=1)
    
    # Calculate movement speed based on health and group size
    should_advance = nearby_enemies >= min_group_size
    move_speed = jnp.where(should_advance, 
                            approach_speed * ally_health * health_aggression_scale,
                            -retreat_speed * ally_health)  # Scale retreat speed with health
    
    # Add combat movement
    combat_dx = dx * move_speed
    combat_dy = dy * move_speed
    
    # Combine all forces
    dx = formation_dx + velocity_match_x + combat_dx
    dy = formation_dy + velocity_match_y + combat_dy
    
    # Apply damping
    dx = dx - ally_vx * damping
    dy = dy - ally_vy * damping
    
    return dx, dy
