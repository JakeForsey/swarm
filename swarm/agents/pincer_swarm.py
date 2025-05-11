import jax
import jax.numpy as jnp

base_radius: float = 0.15
rotation_speed: float = 0.25
pincer_angle: float = 0.3
formation_weight: float = 1.2
velocity_weight: float = 0.1

chase_radius: float = 0.3
chase_weight: float = 0.15
min_group_size: int = 2
health_aggression_scale: float = 1.3
perception_radius: float = 0.35
retreat_speed: float = 0.1

damping: float = 0.08
approach_speed: float = 0.16

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
    batch_size, num_agents = ally_x.shape

    ally_com_x = jnp.mean(ally_x, axis=1, keepdims=True)
    ally_com_y = jnp.mean(ally_y, axis=1, keepdims=True)
    enemy_com_x = jnp.mean(enemy_x, axis=1, keepdims=True)
    enemy_com_y = jnp.mean(enemy_y, axis=1, keepdims=True)

    dx = enemy_com_x - ally_com_x
    dy = enemy_com_y - ally_com_y

    dx = jnp.where(dx > 0.5, dx - 1.0, dx)
    dx = jnp.where(dx < -0.5, dx + 1.0, dx)
    dy = jnp.where(dy > 0.5, dy - 1.0, dy)
    dy = jnp.where(dy < -0.5, dy + 1.0, dy)

    dist = jnp.sqrt(dx**2 + dy**2) + 1e-5
    dx = dx / dist
    dy = dy / dist

    base_angle = jnp.arctan2(dy, dx)

    rotation = t * rotation_speed
    rotation = jnp.reshape(rotation, (batch_size, 1))

    half_agents = num_agents // 2
    angles1 = jnp.linspace(-jnp.pi/2, jnp.pi/2, half_agents)
    angles2 = jnp.linspace(jnp.pi/2, 3*jnp.pi/2, num_agents - half_agents)

    angles1 = jnp.reshape(angles1, (1, half_agents))
    angles2 = jnp.reshape(angles2, (1, num_agents - half_agents))
    
    angles1 = angles1 + rotation + base_angle
    angles2 = angles2 + rotation + base_angle + jnp.pi * pincer_angle
    
    target_x1 = ally_com_x + base_radius * jnp.cos(angles1)
    target_y1 = ally_com_y + base_radius * jnp.sin(angles1)
    target_x2 = ally_com_x + base_radius * jnp.cos(angles2)
    target_y2 = ally_com_y + base_radius * jnp.sin(angles2)
    
    target_x = jnp.concatenate([target_x1, target_x2], axis=1)
    target_y = jnp.concatenate([target_y1, target_y2], axis=1)
    
    formation_dx = (target_x - ally_x) * formation_weight
    formation_dy = (target_y - ally_y) * formation_weight
    
    target_vx1 = -base_radius * jnp.sin(angles1) * rotation_speed
    target_vy1 = base_radius * jnp.cos(angles1) * rotation_speed
    target_vx2 = -base_radius * jnp.sin(angles2) * rotation_speed
    target_vy2 = base_radius * jnp.cos(angles2) * rotation_speed
    
    target_vx = jnp.concatenate([target_vx1, target_vx2], axis=1)
    target_vy = jnp.concatenate([target_vy1, target_vy2], axis=1)
    
    velocity_match_x = (target_vx - ally_vx) * velocity_weight
    velocity_match_y = (target_vy - ally_vy) * velocity_weight
    
    enemy_dx = ally_x[:, None, :] - enemy_x[:, :, None]
    enemy_dy = ally_y[:, None, :] - enemy_y[:, :, None]
    
    enemy_dx = jnp.where(enemy_dx > 0.5, enemy_dx - 1.0, enemy_dx)
    enemy_dx = jnp.where(enemy_dx < -0.5, enemy_dx + 1.0, enemy_dx)
    enemy_dy = jnp.where(enemy_dy > 0.5, enemy_dy - 1.0, enemy_dy)
    enemy_dy = jnp.where(enemy_dy < -0.5, enemy_dy + 1.0, enemy_dy)
    
    enemy_dist = jnp.sqrt(enemy_dx**2 + enemy_dy**2)
    
    nearby_enemies = jnp.sum(enemy_dist < chase_radius, axis=1)
    
    should_advance = nearby_enemies >= min_group_size
    move_speed = jnp.where(should_advance, 
                            approach_speed * ally_health * health_aggression_scale,
                            -retreat_speed * ally_health)

    combat_dx = dx * move_speed
    combat_dy = dy * move_speed

    dx = formation_dx + velocity_match_x + combat_dx
    dy = formation_dy + velocity_match_y + combat_dy

    dx = dx - ally_vx * damping
    dy = dy - ally_vy * damping

    return dx, dy
