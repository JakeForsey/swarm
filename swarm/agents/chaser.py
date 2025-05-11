import jax
import jax.numpy as jnp

CHASE_RADIUS = 0.3
CHASE_WEIGHT = 0.01
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
    """Chaser agent that aggressively pursues and attacks enemies."""
    dx = ally_x[:, None, :] - enemy_x[:, :, None]
    dy = ally_y[:, None, :] - enemy_y[:, :, None]
    dist = jnp.sqrt(dx**2 + dy**2)

    x_action = jnp.zeros_like(ally_vx)
    y_action = jnp.zeros_like(ally_vy)

    min_dist = jnp.min(dist, axis=1)
    closest_target_idx = jnp.argmin(dist, axis=1)
    
    batch_idx = jnp.arange(ally_x.shape[0])[:, None]
    target_idx = closest_target_idx
    agent_idx = jnp.arange(ally_x.shape[1])[None, :]
    
    closest_dx = dx[batch_idx, target_idx, agent_idx]
    closest_dy = dy[batch_idx, target_idx, agent_idx]
    
    chase_mask = min_dist < CHASE_RADIUS
    x_action += -closest_dx * chase_mask * CHASE_WEIGHT  # Move towards target
    y_action += -closest_dy * chase_mask * CHASE_WEIGHT

    xkey, ykey, _ = jax.random.split(key, 3)
    x_action += jax.random.uniform(xkey, ally_x.shape, minval=-RANDOM_WEIGHT, maxval=RANDOM_WEIGHT)
    y_action += jax.random.uniform(ykey, ally_y.shape, minval=-RANDOM_WEIGHT, maxval=RANDOM_WEIGHT)

    return x_action, y_action
