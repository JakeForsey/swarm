import jax
import jax.numpy as jnp
from typing import Tuple

base_radius=0.18795883994808754
rotation_speed=0.27576399081988195
spiral_tightness=0.10301011198249985
formation_weight=0.141510346628356
velocity_weight=0.05460453492299239
chase_radius=0.357469506800012
chase_weight=0.016298297483683093
min_group_size=1
health_aggression_scale=1.0762224852357951
perception_radius=0.36716339620958827
damping=0.10547655251076879
approach_speed=0.1658056010939894

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
    """Create a spiral swarm agent with dynamic center and combat behavior."""
    # Calculate center of mass for both teams
    ally_com_x = jnp.mean(ally_x, axis=1, keepdims=True)
    ally_com_y = jnp.mean(ally_y, axis=1, keepdims=True)
    enemy_com_x = jnp.mean(enemy_x, axis=1, keepdims=True)
    enemy_com_y = jnp.mean(enemy_y, axis=1, keepdims=True)

    # Calculate direction to enemy center
    dx_to_enemy = enemy_com_x - ally_com_x
    dy_to_enemy = enemy_com_y - ally_com_y
    dist_to_enemy = jnp.sqrt(dx_to_enemy**2 + dy_to_enemy**2)
    
    # Handle wrapping for enemy direction
    dx_to_enemy = jnp.where(dx_to_enemy > 0.5, dx_to_enemy - 1.0, dx_to_enemy)
    dx_to_enemy = jnp.where(dx_to_enemy < -0.5, dx_to_enemy + 1.0, dx_to_enemy)
    dy_to_enemy = jnp.where(dy_to_enemy > 0.5, dy_to_enemy - 1.0, dy_to_enemy)
    dy_to_enemy = jnp.where(dy_to_enemy < -0.5, dy_to_enemy + 1.0, dy_to_enemy)
    
    # Normalize direction to enemy
    dx_to_enemy = dx_to_enemy / (dist_to_enemy + 1e-5)
    dy_to_enemy = dy_to_enemy / (dist_to_enemy + 1e-5)

    # Calculate relative positions to center
    rel_x = ally_x - ally_com_x
    rel_y = ally_y - ally_com_y
    
    # Handle wrapping for relative positions
    rel_x = jnp.where(rel_x > 0.5, rel_x - 1.0, rel_x)
    rel_x = jnp.where(rel_x < -0.5, rel_x + 1.0, rel_x)
    rel_y = jnp.where(rel_y > 0.5, rel_y - 1.0, rel_y)
    rel_y = jnp.where(rel_y < -0.5, rel_y + 1.0, rel_y)
    
    rel_dist = jnp.sqrt(rel_x**2 + rel_y**2)
    
    # Calculate angles for spiral formation
    angles = jnp.arctan2(rel_y, rel_x)
    
    # Calculate target positions in spiral
    # Spiral tightness increases with distance from center
    spiral_factor = 1.0 + spiral_tightness * (rel_dist / base_radius)
    # Ensure proper broadcasting by reshaping time
    time_factor = jnp.reshape(t, (-1, 1)) * rotation_speed
    target_angles = angles + time_factor * spiral_factor
    
    # Calculate target positions
    target_x = ally_com_x + base_radius * jnp.cos(target_angles)
    target_y = ally_com_y + base_radius * jnp.sin(target_angles)
    
    # Add movement toward enemy
    target_x += dx_to_enemy * approach_speed
    target_y += dy_to_enemy * approach_speed
    
    # Calculate target velocities (tangential to the spiral)
    target_vx = -base_radius * jnp.sin(target_angles) * rotation_speed
    target_vy = base_radius * jnp.cos(target_angles) * rotation_speed
    
    # Calculate formation movement
    formation_dx = (target_x - ally_x) * formation_weight
    formation_dy = (target_y - ally_y) * formation_weight
    
    # Calculate velocity matching
    velocity_match_x = (target_vx - ally_vx) * velocity_weight
    velocity_match_y = (target_vy - ally_vy) * velocity_weight
    
    # Initialize combat forces
    combat_dx = jnp.zeros_like(ally_x)
    combat_dy = jnp.zeros_like(ally_y)
    
    # Calculate distances to enemies
    enemy_dx = ally_x[:, None, :] - enemy_x[:, :, None]
    enemy_dy = ally_y[:, None, :] - enemy_y[:, :, None]
    
    # Handle wrapping for enemy distances
    enemy_dx = jnp.where(enemy_dx > 0.5, enemy_dx - 1.0, enemy_dx)
    enemy_dx = jnp.where(enemy_dx < -0.5, enemy_dx + 1.0, enemy_dx)
    enemy_dy = jnp.where(enemy_dy > 0.5, enemy_dy - 1.0, enemy_dy)
    enemy_dy = jnp.where(enemy_dy < -0.5, enemy_dy + 1.0, enemy_dy)
    
    enemy_dist = jnp.sqrt(enemy_dx**2 + enemy_dy**2)
    
    # Find closest enemy for each agent
    min_enemy_dist = jnp.min(enemy_dist, axis=1)
    closest_enemy_idx = jnp.argmin(enemy_dist, axis=1)
    
    # Get relative positions to closest enemies
    batch_idx = jnp.arange(ally_x.shape[0])[:, None]
    enemy_idx = closest_enemy_idx
    agent_idx = jnp.arange(ally_x.shape[1])[None, :]
    
    closest_enemy_dx = enemy_dx[batch_idx, enemy_idx, agent_idx]
    closest_enemy_dy = enemy_dy[batch_idx, enemy_idx, agent_idx]
    
    # Calculate aggression based on health and group size
    health_aggression = ally_health * health_aggression_scale
    group_size = jnp.sum(enemy_dist < perception_radius, axis=1)
    group_advantage = group_size > min_group_size
    
    # Chase if we have group advantage and enemy is within range
    chase_mask = (min_enemy_dist < chase_radius) & group_advantage
    chase_strength = chase_mask * health_aggression
    
    # Add combat movement
    combat_dx = -closest_enemy_dx * chase_strength * chase_weight
    combat_dy = -closest_enemy_dy * chase_strength * chase_weight
    
    # Combine all forces
    dx = formation_dx + velocity_match_x + combat_dx
    dy = formation_dy + velocity_match_y + combat_dy
    
    # Apply damping
    dx -= ally_vx * damping
    dy -= ally_vy * damping
    
    return dx, dy
