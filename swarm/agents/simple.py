from typing import Tuple

import jax
import jax.numpy as jnp

ATTACK_THRESHOLD = -0.00791715
ATTACK_SPEED = 2.995118
FLEE_SPEED = 0.13312322

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
    batch_size, num_allies = ally_x.shape
    
    dx = ally_x[:, :, None] - enemy_x[:, None, :]
    dy = ally_y[:, :, None] - enemy_y[:, None, :]
    
    # Handle wrapping
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
