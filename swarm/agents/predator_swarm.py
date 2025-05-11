import jax
import jax.numpy as jnp

formation_scale: float = 0.3
formation_weight: float = 0.8

attack_speed: float = 0.2
pursue_speed: float = 0.15
retreat_speed: float = 0.1

focus_fire_radius: float = 0.2
bait_threshold: float = 0.9
surround_radius: float = 0.4

engage_threshold: float = 0.7
disengage_threshold: float = 0.5

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
    enemy_total_health = enemy_health
    primary_target = jnp.argmin(enemy_total_health, axis=1, keepdims=True)

    target_x = jnp.take_along_axis(enemy_x, primary_target, axis=1)
    target_y = jnp.take_along_axis(enemy_y, primary_target, axis=1)
    
    dx_to_target = target_x - ally_x
    dy_to_target = target_y - ally_y
    
    is_bait = ally_health > bait_threshold
    
    num_agents = ally_x.shape[1]
    angles = jnp.linspace(0, 2*jnp.pi, num_agents, endpoint=False)
    
    surround_x = surround_radius * jnp.cos(angles)
    surround_y = surround_radius * jnp.sin(angles)
    
    surround_x = target_x + surround_x[None, :]
    surround_y = target_y + surround_y[None, :]

    dx = jnp.where(is_bait, 
                    dx_to_target,
                    surround_x - ally_x)
    dy = jnp.where(is_bait,
                    dy_to_target,
                    surround_y - ally_y)
    
    magnitude = jnp.sqrt(dx**2 + dy**2) + 1e-10
    dx = dx / magnitude
    dy = dy / magnitude
    
    speed = jnp.where(
        is_bait,
        jnp.where(ally_health > engage_threshold,
                    attack_speed,
                    retreat_speed),
        pursue_speed
    )
    
    dx = dx * speed
    dy = dy * speed
    
    center_x = jnp.mean(ally_x, axis=1, keepdims=True)
    center_y = jnp.mean(ally_y, axis=1, keepdims=True)
    dx = dx + formation_weight * (center_x - ally_x)
    dy = dy + formation_weight * (center_y - ally_y)
    
    return dx, dy
