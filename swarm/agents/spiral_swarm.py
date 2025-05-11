import jax
import jax.numpy as jnp

base_radius=0.187
rotation_speed=0.275
spiral_tightness=0.103
formation_weight=0.141
velocity_weight=0.054
chase_radius=0.357
chase_weight=0.016
min_group_size=1
health_aggression_scale=1.076
perception_radius=0.367
damping=0.105
approach_speed=0.165

@jax.jit
def act(
    t,
    key,
    ally_x,
    ally_y,
    ally_vx,
    ally_vy,
    ally_health,
    enemy_y,
    enemy_x,
    enemy_vx,
    enemy_vy,
    enemy_health,
):
    """Create a spiral swarm agent with dynamic center and combat behavior."""
    ally_com_x = jnp.mean(ally_x, axis=1, keepdims=True)
    ally_com_y = jnp.mean(ally_y, axis=1, keepdims=True)
    enemy_com_x = jnp.mean(enemy_x, axis=1, keepdims=True)
    enemy_com_y = jnp.mean(enemy_y, axis=1, keepdims=True)

    dx_to_enemy = enemy_com_x - ally_com_x
    dy_to_enemy = enemy_com_y - ally_com_y
    dist_to_enemy = jnp.sqrt(dx_to_enemy**2 + dy_to_enemy**2)
    
    dx_to_enemy = jnp.where(dx_to_enemy > 0.5, dx_to_enemy - 1.0, dx_to_enemy)
    dx_to_enemy = jnp.where(dx_to_enemy < -0.5, dx_to_enemy + 1.0, dx_to_enemy)
    dy_to_enemy = jnp.where(dy_to_enemy > 0.5, dy_to_enemy - 1.0, dy_to_enemy)
    dy_to_enemy = jnp.where(dy_to_enemy < -0.5, dy_to_enemy + 1.0, dy_to_enemy)
    
    dx_to_enemy = dx_to_enemy / (dist_to_enemy + 1e-5)
    dy_to_enemy = dy_to_enemy / (dist_to_enemy + 1e-5)

    rel_x = ally_x - ally_com_x
    rel_y = ally_y - ally_com_y
    
    rel_x = jnp.where(rel_x > 0.5, rel_x - 1.0, rel_x)
    rel_x = jnp.where(rel_x < -0.5, rel_x + 1.0, rel_x)
    rel_y = jnp.where(rel_y > 0.5, rel_y - 1.0, rel_y)
    rel_y = jnp.where(rel_y < -0.5, rel_y + 1.0, rel_y)
    
    rel_dist = jnp.sqrt(rel_x**2 + rel_y**2)
    
    angles = jnp.arctan2(rel_y, rel_x)
    
    spiral_factor = 1.0 + spiral_tightness * (rel_dist / base_radius)
    time_factor = jnp.reshape(t, (-1, 1)) * rotation_speed
    target_angles = angles + time_factor * spiral_factor
    
    target_x = ally_com_x + base_radius * jnp.cos(target_angles)
    target_y = ally_com_y + base_radius * jnp.sin(target_angles)
    
    target_x += dx_to_enemy * approach_speed
    target_y += dy_to_enemy * approach_speed
    
    target_vx = -base_radius * jnp.sin(target_angles) * rotation_speed
    target_vy = base_radius * jnp.cos(target_angles) * rotation_speed
    
    formation_dx = (target_x - ally_x) * formation_weight
    formation_dy = (target_y - ally_y) * formation_weight
    
    velocity_match_x = (target_vx - ally_vx) * velocity_weight
    velocity_match_y = (target_vy - ally_vy) * velocity_weight
    
    combat_dx = jnp.zeros_like(ally_x)
    combat_dy = jnp.zeros_like(ally_y)
    
    enemy_dx = ally_x[:, None, :] - enemy_x[:, :, None]
    enemy_dy = ally_y[:, None, :] - enemy_y[:, :, None]
    
    enemy_dx = jnp.where(enemy_dx > 0.5, enemy_dx - 1.0, enemy_dx)
    enemy_dx = jnp.where(enemy_dx < -0.5, enemy_dx + 1.0, enemy_dx)
    enemy_dy = jnp.where(enemy_dy > 0.5, enemy_dy - 1.0, enemy_dy)
    enemy_dy = jnp.where(enemy_dy < -0.5, enemy_dy + 1.0, enemy_dy)
    
    enemy_dist = jnp.sqrt(enemy_dx**2 + enemy_dy**2)
    
    min_enemy_dist = jnp.min(enemy_dist, axis=1)
    closest_enemy_idx = jnp.argmin(enemy_dist, axis=1)
    
    batch_idx = jnp.arange(ally_x.shape[0])[:, None]
    enemy_idx = closest_enemy_idx
    agent_idx = jnp.arange(ally_x.shape[1])[None, :]
    
    closest_enemy_dx = enemy_dx[batch_idx, enemy_idx, agent_idx]
    closest_enemy_dy = enemy_dy[batch_idx, enemy_idx, agent_idx]
    
    health_aggression = ally_health * health_aggression_scale
    group_size = jnp.sum(enemy_dist < perception_radius, axis=1)
    group_advantage = group_size > min_group_size
    
    chase_mask = (min_enemy_dist < chase_radius) & group_advantage
    chase_strength = chase_mask * health_aggression
    
    combat_dx = -closest_enemy_dx * chase_strength * chase_weight
    combat_dy = -closest_enemy_dy * chase_strength * chase_weight
    
    dx = formation_dx + velocity_match_x + combat_dx
    dy = formation_dy + velocity_match_y + combat_dy
    
    dx -= ally_vx * damping
    dy -= ally_vy * damping
    
    return dx, dy
