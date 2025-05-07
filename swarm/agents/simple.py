from typing import Tuple

import jax
import jax.numpy as jnp

from swarm.env import State

SPEED = 0.2

def act(state: State, team: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if team == 1:
        ally_x, ally_y = state.x1, state.y1
        ally_health = state.health1
        enemy_x, enemy_y = state.x2, state.y2
        enemy_health = state.health2
    else:
        ally_x, ally_y = state.x2, state.y2
        ally_health = state.health2
        enemy_x, enemy_y = state.x1, state.y1
        enemy_health = state.health1
    return _act(ally_x, ally_y, ally_health, enemy_x, enemy_y, enemy_health)

@jax.jit
def _act(
    ally_x: jnp.ndarray, ally_y: jnp.ndarray,
    ally_health: jnp.ndarray,
    enemy_x: jnp.ndarray, enemy_y: jnp.ndarray,
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
    should_attack = health_diff >= 0
    
    nearest_enemy_dx = dx[batch_idx, agent_idx, nearest_enemy_idx]
    nearest_enemy_dy = dy[batch_idx, agent_idx, nearest_enemy_idx]
    
    move_speed = jnp.where(should_attack, -SPEED, SPEED)
    dx = nearest_enemy_dx * move_speed
    dy = nearest_enemy_dy * move_speed
            
    return dx, dy
