from typing import Tuple

import jax
import jax.numpy as jnp

from swarm.env import State

def act(state: State, team: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Moves agents in pairs with coordinated random movement and convergence."""
    if team == 1:
        ally_x, ally_y = state.x1, state.y1
        vx, vy = state.vx1, state.vy1
        ally_health = state.health1
        enemy_x, enemy_y = state.x2, state.y2
        enemy_health = state.health2
    else:
        ally_x, ally_y = state.x2, state.y2
        vx, vy = state.vx2, state.vy2
        ally_health = state.health2
        enemy_x, enemy_y = state.x1, state.y1
        enemy_health = state.health1

    dx, dy = _act(ally_x, ally_y, ally_health, enemy_x, enemy_y, enemy_health)
    return dx - vx, dy - vy

@jax.jit
def _act(
    ally_x: jnp.ndarray,
    ally_y: jnp.ndarray,
    ally_health: jnp.ndarray,
    enemy_x: jnp.ndarray,
    enemy_y: jnp.ndarray,
    enemy_health: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    batch_size, num_allies = ally_x.shape
    num_pairs = num_allies // 2
    
    left_agents_x = ally_x[:, :num_pairs]
    right_agents_x = ally_x[:, num_pairs:]
    left_agents_y = ally_y[:, :num_pairs]
    right_agents_y = ally_y[:, num_pairs:]
    left_agents_health = ally_health[:, :num_pairs]
    right_agents_health = ally_health[:, num_pairs:]
    pair_health = left_agents_health + right_agents_health

    pairs_dx = left_agents_x - right_agents_x
    pairs_dy = left_agents_y - right_agents_y
    pairs_distances = jnp.sqrt(pairs_dx**2 + pairs_dy**2)
    in_pair = jnp.concatenate([pairs_distances, pairs_distances], axis=1) < 0.02

    pairs_mid_x = (left_agents_x + right_agents_x) / 2
    pairs_mid_y = (left_agents_y + right_agents_y) / 2

    nearest_enemy_x = enemy_x[:, None, :] - pairs_mid_x[:, :, None]
    nearest_enemy_y = enemy_y[:, None, :] - pairs_mid_y[:, :, None]
    nearest_enemy_distances = jnp.sqrt(nearest_enemy_x**2 + nearest_enemy_y**2)
    nearest_enemy_index = jnp.argmin(nearest_enemy_distances, axis=2)
    nearest_enemy_x = jnp.take_along_axis(enemy_x, nearest_enemy_index, axis=1)
    nearest_enemy_y = jnp.take_along_axis(enemy_y, nearest_enemy_index, axis=1)
    nearest_enemy_health = jnp.take_along_axis(enemy_health, nearest_enemy_index, axis=1)

    should_attack = pair_health >= nearest_enemy_health
    combat_dx = jnp.where(
        should_attack, 
        nearest_enemy_x - pairs_mid_x,
        pairs_mid_x - nearest_enemy_x,

    )
    combat_dy = jnp.where(
        should_attack, 
        nearest_enemy_y - pairs_mid_y,
        pairs_mid_y - nearest_enemy_y,
    )

    dx = jnp.where(
        in_pair,
        jnp.concatenate([combat_dx, combat_dx], axis=1),
        jnp.concatenate([-pairs_dx, pairs_dx], axis=1),
    )
    dy = jnp.where(
        in_pair,
        jnp.concatenate([combat_dy, combat_dy], axis=1),
        jnp.concatenate([-pairs_dy, pairs_dy], axis=1),
    )

    return dx, dy
