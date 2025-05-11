import jax
import jax.numpy as jnp

ATTACK_THRESHOLD = -0.00791715
ATTACK_SPEED = 2.995118
FLEE_SPEED = 0.13312322

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
    batch_size, num_allies = ally_x.shape
    
    dx = ally_x[:, :, None] - enemy_x[:, None, :]
    dy = ally_y[:, :, None] - enemy_y[:, None, :]
    
    dx = jnp.where(dx > 0.5, dx - 1.0, dx)
    dx = jnp.where(dx < -0.5, dx + 1.0, dx)
    dy = jnp.where(dy > 0.5, dy - 1.0, dy)
    dy = jnp.where(dy < -0.5, dy + 1.0, dy)
    
    alive_enemies = enemy_health > 0
    distances = jnp.sqrt(dx**2 + dy**2) + 1e-5
    distances = jnp.where(alive_enemies[:, None, :], distances, jnp.inf)
    
    nearest_enemy_idx = jnp.argmin(distances, axis=2)
    batch_idx = jnp.arange(batch_size)[:, None]
    agent_idx = jnp.arange(num_allies)[None, :]
    nearest_enemy_health = enemy_health[batch_idx, nearest_enemy_idx]
    
    health_diff = ally_health - nearest_enemy_health
    should_attack = health_diff >= ATTACK_THRESHOLD
    
    nearest_enemy_dx = dx[batch_idx, agent_idx, nearest_enemy_idx]
    nearest_enemy_dy = dy[batch_idx, agent_idx, nearest_enemy_idx]
    
    move_speed = jnp.where(should_attack, -ATTACK_SPEED, FLEE_SPEED)

    dx_out = nearest_enemy_dx * move_speed 
    dy_out = nearest_enemy_dy * move_speed
            
    return dx_out, dy_out
