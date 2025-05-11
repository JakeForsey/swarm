import jax
import jax.numpy as jnp

COHESION_RADIUS = 0.2
FLEE_RADIUS = 0.3
FLEE_WEIGHT = 0.01
COHESION_WEIGHT = 0.005
RANDOM_WEIGHT = 0.001

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
    """Fleeing agent that avoids enemies and maintains distance."""
    x_action = jnp.zeros_like(ally_vx)
    y_action = jnp.zeros_like(ally_vy)

    enemy_dx = ally_x[:, None, :] - enemy_x[:, :, None]
    enemy_dy = ally_y[:, None, :] - enemy_y[:, :, None]
    enemy_dist = jnp.sqrt(enemy_dx**2 + enemy_dy**2)

    min_enemy_dist = jnp.min(enemy_dist, axis=1)
    closest_enemy_idx = jnp.argmin(enemy_dist, axis=1)
    
    batch_idx = jnp.arange(ally_x.shape[0])[:, None]
    enemy_idx = closest_enemy_idx
    agent_idx = jnp.arange(ally_x.shape[1])[None, :]
    
    closest_enemy_dx = enemy_dx[batch_idx, enemy_idx, agent_idx]
    closest_enemy_dy = enemy_dy[batch_idx, enemy_idx, agent_idx]
    
    flee_mask = min_enemy_dist < FLEE_RADIUS
    x_action += closest_enemy_dx * flee_mask * FLEE_WEIGHT
    y_action += closest_enemy_dy * flee_mask * FLEE_WEIGHT

    ally_dx = ally_x[:, None, :] - ally_x[:, :, None]
    ally_dy = ally_y[:, None, :] - ally_y[:, :, None]
    ally_dist = jnp.sqrt(ally_dx**2 + ally_dy**2)

    nearby_mask = (ally_dist < COHESION_RADIUS) & (ally_dist > 0)
    x_total = jnp.sum(ally_x[:, None, :] * nearby_mask, axis=1)
    y_total = jnp.sum(ally_y[:, None, :] * nearby_mask, axis=1)
    nearby_count = jnp.sum(nearby_mask, axis=1)
    
    x_avg = jnp.where(nearby_count > 0, x_total / nearby_count, ally_x)
    y_avg = jnp.where(nearby_count > 0, y_total / nearby_count, ally_y)
    
    x_action += (x_avg - ally_x) * COHESION_WEIGHT
    y_action += (y_avg - ally_y) * COHESION_WEIGHT

    xkey, ykey, _ = jax.random.split(key, 3)
    x_action += jax.random.uniform(xkey, ally_x.shape, minval=-RANDOM_WEIGHT, maxval=RANDOM_WEIGHT)
    y_action += jax.random.uniform(ykey, ally_y.shape, minval=-RANDOM_WEIGHT, maxval=RANDOM_WEIGHT)

    return x_action, y_action
