import jax
import jax.numpy as jnp

outer_radius: float = 0.2
inner_radius: float = 0.08
formation_weight: float = 0.8

patrol_speed: float = 0.12
retreat_speed: float = 0.15
return_speed: float = 0.1

retreat_threshold: float = 0.7
return_threshold: float = 0.9

attack_range: float = 0.15
rotation_speed: float = 0.1

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
    center_x = jnp.mean(ally_x, axis=1, keepdims=True)
    center_y = jnp.mean(ally_y, axis=1, keepdims=True)
    
    needs_healing = ally_health < retreat_threshold
    fully_healed = ally_health > return_threshold
    
    num_agents = ally_x.shape[1]
    base_angles = jnp.linspace(0, 2*jnp.pi, num_agents, endpoint=False)
    rotation = rotation_speed * t[:, None]
    angles = base_angles[None, :] + rotation
    
    ring_x = center_x + outer_radius * jnp.cos(angles)
    ring_y = center_y + outer_radius * jnp.sin(angles)
    
    heal_angles = jnp.linspace(0, 4*jnp.pi, num_agents, endpoint=False)
    heal_radius = jnp.linspace(0, inner_radius, num_agents, endpoint=False)
    heal_x = center_x + heal_radius[None, :] * jnp.cos(heal_angles)[None, :]
    heal_y = center_y + heal_radius[None, :] * jnp.sin(heal_angles)[None, :]
    
    dx_to_enemies = enemy_x[:, :, None] - ally_x[:, None, :]
    dy_to_enemies = enemy_y[:, :, None] - ally_y[:, None, :]
    dist_to_enemies = jnp.sqrt(dx_to_enemies**2 + dy_to_enemies**2)
    nearest_enemy_dist = jnp.min(dist_to_enemies, axis=1)
    
    enemies_in_range = nearest_enemy_dist < attack_range
    
    target_x = jnp.where(needs_healing, heal_x,
                        jnp.where(enemies_in_range, ally_x, ring_x))
    target_y = jnp.where(needs_healing, heal_y,
                        jnp.where(enemies_in_range, ally_y, ring_y))
    
    dx = target_x - ally_x
    dy = target_y - ally_y
    
    magnitude = jnp.sqrt(dx**2 + dy**2) + 1e-10
    dx = dx / magnitude
    dy = dy / magnitude
    
    speed = jnp.where(needs_healing, retreat_speed,
                        jnp.where(fully_healed & ~enemies_in_range, 
                                return_speed,
                                patrol_speed))
    
    dx = dx * speed
    dy = dy * speed
    
    dx = dx + formation_weight * (center_x - ally_x)
    dy = dy + formation_weight * (center_y - ally_y)
    
    return dx, dy
