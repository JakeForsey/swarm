import jax
import jax.numpy as jnp
from typing import Tuple

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
    # Get own and enemy state based on team

    # Calculate center of our formation
    center_x = jnp.mean(ally_x, axis=1, keepdims=True)
    center_y = jnp.mean(ally_y, axis=1, keepdims=True)
    
    # Identify units that need healing
    needs_healing = ally_health < retreat_threshold
    fully_healed = ally_health > return_threshold
    
    # Calculate positions on the defensive ring
    num_agents = ally_x.shape[1]
    # Add rotation based on time
    base_angles = jnp.linspace(0, 2*jnp.pi, num_agents, endpoint=False)
    rotation = rotation_speed * t[:, None]
    angles = base_angles[None, :] + rotation
    
    ring_x = center_x + outer_radius * jnp.cos(angles)
    ring_y = center_y + outer_radius * jnp.sin(angles)
    
    # Calculate positions in the healing zone (spiral pattern)
    heal_angles = jnp.linspace(0, 4*jnp.pi, num_agents, endpoint=False)
    heal_radius = jnp.linspace(0, inner_radius, num_agents, endpoint=False)
    heal_x = center_x + heal_radius[None, :] * jnp.cos(heal_angles)[None, :]
    heal_y = center_y + heal_radius[None, :] * jnp.sin(heal_angles)[None, :]
    
    # Calculate distances to nearest enemy
    dx_to_enemies = enemy_x[:, :, None] - ally_x[:, None, :]
    dy_to_enemies = enemy_y[:, :, None] - ally_y[:, None, :]
    dist_to_enemies = jnp.sqrt(dx_to_enemies**2 + dy_to_enemies**2)
    nearest_enemy_dist = jnp.min(dist_to_enemies, axis=1)
    
    # Determine if enemies are in range
    enemies_in_range = nearest_enemy_dist < attack_range
    
    # Calculate target positions
    # If healing: go to healing zone
    # If healed but enemies nearby: stay in position
    # If healed and safe: return to ring
    target_x = jnp.where(needs_healing, heal_x,
                        jnp.where(enemies_in_range, ally_x, ring_x))
    target_y = jnp.where(needs_healing, heal_y,
                        jnp.where(enemies_in_range, ally_y, ring_y))
    
    # Calculate move directions
    dx = target_x - ally_x
    dy = target_y - ally_y
    
    # Normalize directions
    magnitude = jnp.sqrt(dx**2 + dy**2) + 1e-10
    dx = dx / magnitude
    dy = dy / magnitude
    
    # Adjust speed based on state
    speed = jnp.where(needs_healing, retreat_speed,
                        jnp.where(fully_healed & ~enemies_in_range, 
                                return_speed,
                                patrol_speed))
    
    # Apply speed to movement
    dx = dx * speed
    dy = dy * speed
    
    # Add formation cohesion
    dx = dx + formation_weight * (center_x - ally_x)
    dy = dy + formation_weight * (center_y - ally_y)
    
    return dx, dy
