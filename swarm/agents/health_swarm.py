import jax
import jax.numpy as jnp

attack_speed: float = 0.2
flee_speed: float = 0.15

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
    batch_size = ally_x.shape[0]
    num_allies = ally_x.shape[1]

    dx = ally_x[:, :, None] - enemy_x[:, None, :]
    dy = ally_y[:, :, None] - enemy_y[:, None, :]
    
    dx = jnp.where(dx > 0.5, dx - 1.0, dx)
    dx = jnp.where(dx < -0.5, dx + 1.0, dx)
    dy = jnp.where(dy > 0.5, dy - 1.0, dy)
    dy = jnp.where(dy < -0.5, dy + 1.0, dy)
    
    distances = jnp.sqrt(dx**2 + dy**2) + 1e-5
    alive_enemies = enemy_health > 0
    distances = jnp.where(alive_enemies[:, None, :], distances, jnp.inf)
    
    nearest_enemy_idx = jnp.argmin(distances, axis=2)
    batch_idx = jnp.arange(batch_size)[:, None]
    agent_idx = jnp.arange(num_allies)[None, :]
    nearest_enemy_health = enemy_health[batch_idx, nearest_enemy_idx]
    
    health_diff = ally_health - nearest_enemy_health
    should_attack = health_diff >= 0
    
    nearest_enemy_dx = dx[batch_idx, agent_idx, nearest_enemy_idx]
    nearest_enemy_dy = dy[batch_idx, agent_idx, nearest_enemy_idx]
    
    move_speed = jnp.where(should_attack, -attack_speed, flee_speed)
    dx = nearest_enemy_dx * move_speed
    dy = nearest_enemy_dy * move_speed

    return dx, dy
