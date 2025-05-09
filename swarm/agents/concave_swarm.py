import jax
import jax.numpy as jnp
from typing import Tuple

LINE_LENGTH: float = 0.32  # Balanced formation tightness
CURVE_STRENGTH: float = 0.18  # Moderate curve for stability
FORMATION_WEIGHT: float = 1.1  # Balanced formation priority

# Movement parameters
ADVANCE_SPEED: float = 0.16  # Balanced advance speed
RETREAT_SPEED: float = 0.19  # Balanced retreat speed
DAMPING: float = 0.07  # Balanced damping

# Combat parameters
ENGAGEMENT_DISTANCE: float = 0.28  # Balanced engagement range
MIN_GROUP_SIZE: int = 2  # Moderate aggression
HEALTH_AGGRESSION: float = 1.35  # Balanced aggression scaling

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
    state_t = t
    # Get team positions and velocities
    batch_size = ally_x.shape[0]
    num_agents = ally_x.shape[1]

    # Calculate centers with explicit shapes
    ally_com_x = jnp.mean(ally_x, axis=1, keepdims=True)   # Shape: (batch_size, 1)
    ally_com_y = jnp.mean(ally_y, axis=1, keepdims=True)   # Shape: (batch_size, 1)
    enemy_com_x = jnp.mean(enemy_x, axis=1, keepdims=True) # Shape: (batch_size, 1)
    enemy_com_y = jnp.mean(enemy_y, axis=1, keepdims=True) # Shape: (batch_size, 1)

    # Calculate direction to enemy with explicit shapes
    dx = enemy_com_x - ally_com_x  # Shape: (batch_size, 1)
    dy = enemy_com_y - ally_com_y  # Shape: (batch_size, 1)
        
    # Handle wrapping
    dx = jnp.where(dx > 0.5, dx - 1.0, dx)
    dx = jnp.where(dx < -0.5, dx + 1.0, dx)
    dy = jnp.where(dy > 0.5, dy - 1.0, dy)
    dy = jnp.where(dy < -0.5, dy + 1.0, dy)
    
    # Normalize direction
    dist = jnp.sqrt(dx**2 + dy**2) + 1e-5
    dx = dx / dist  # Shape: (batch_size, 1)
    dy = dy / dist  # Shape: (batch_size, 1)
    
    # Create line positions
    t = jnp.linspace(-1.0, 1.0, num_agents)  # Shape: (num_agents,)    
    t = jnp.tile(t[None, :], (batch_size, 1))  # Shape: (batch_size, num_agents)

    # Calculate perpendicular direction with explicit reshaping
    perp_x = -dy  # Shape should be (batch_size, 1)
    perp_y = dx   # Shape should be (batch_size, 1)
    
    # Time-based formation scaling with explicit shape
    formation_scale = jnp.minimum(1.0, 1.0 + state_t * 0.1)
    # Reshape formation_scale to (batch_size, 1) for broadcasting
    formation_scale = jnp.reshape(formation_scale, (batch_size, 1))
    
    # Calculate base line positions with explicit broadcasting
    line_x = jnp.multiply(t, perp_x)  # First multiply
    line_x = jnp.multiply(line_x, LINE_LENGTH)  # Second multiply
    line_x = jnp.multiply(line_x, formation_scale)  # Third multiply
    
    line_y = jnp.multiply(t, perp_y)
    line_y = jnp.multiply(line_y, LINE_LENGTH)
    line_y = jnp.multiply(line_y, formation_scale)
        
    # Add curve toward enemy
    curve = CURVE_STRENGTH * (1.0 - t**2)  # Shape: (batch_size, num_agents)
    
    # Ensure dx and dy are properly shaped for broadcasting
    dx_broad = jnp.broadcast_to(dx, (batch_size, num_agents))
    dy_broad = jnp.broadcast_to(dy, (batch_size, num_agents))
    
    curve_x = jnp.multiply(curve, dx_broad)  # Explicit multiply
    curve_x = jnp.multiply(curve_x, LINE_LENGTH)
    curve_x = jnp.multiply(curve_x, formation_scale)
    
    curve_y = jnp.multiply(curve, dy_broad)
    curve_y = jnp.multiply(curve_y, LINE_LENGTH)
    curve_y = jnp.multiply(curve_y, formation_scale)
        
    # Combine for final target positions
    target_x = ally_com_x + line_x + curve_x  # Shape: (batch_size, num_agents)
    target_y = ally_com_y + line_y + curve_y  # Shape: (batch_size, num_agents)
        
    # Calculate formation forces
    formation_dx = (target_x - ally_x) * FORMATION_WEIGHT
    formation_dy = (target_y - ally_y) * FORMATION_WEIGHT
    
    # Calculate combat distances
    enemy_dx = ally_x[:, None, :] - enemy_x[:, :, None]  # Shape: (batch_size, num_enemies, num_agents)
    enemy_dy = ally_y[:, None, :] - enemy_y[:, :, None]
    
    # Handle wrapping for combat
    enemy_dx = jnp.where(enemy_dx > 0.5, enemy_dx - 1.0, enemy_dx)
    enemy_dx = jnp.where(enemy_dx < -0.5, enemy_dx + 1.0, enemy_dx)
    enemy_dy = jnp.where(enemy_dy > 0.5, enemy_dy - 1.0, enemy_dy)
    enemy_dy = jnp.where(enemy_dy < -0.5, enemy_dy + 1.0, enemy_dy)
    
    enemy_dist = jnp.sqrt(enemy_dx**2 + enemy_dy**2)
    
    # Count nearby enemies
    nearby_enemies = jnp.sum(enemy_dist < ENGAGEMENT_DISTANCE, axis=1)
    
    # Calculate movement speed
    should_advance = nearby_enemies >= MIN_GROUP_SIZE
    move_speed = jnp.where(should_advance, 
                            ADVANCE_SPEED * ally_health * HEALTH_AGGRESSION,
                            -RETREAT_SPEED)
    
    # Add combat movement (ensure proper shapes for broadcasting)
    combat_dx = dx.reshape(batch_size, 1) * move_speed
    combat_dy = dy.reshape(batch_size, 1) * move_speed

    # Combine forces
    dx = formation_dx + 0.3 * combat_dx
    dy = formation_dy + 0.3 * combat_dy
    
    # Apply damping
    dx = dx - ally_vx * DAMPING
    dy = dy - ally_vy * DAMPING
        
    return dx, dy
