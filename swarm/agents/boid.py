import jax
import jax.numpy as jnp

SEPERATION_RADIUS = 0.1
PERCEPTION_RADIUS = 0.3
SEPERATION_WEIGHT = 0.001
ALIGNMENT_WEIGHT = 0.001
COHESION_WEIGHT = 0.01

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
    """Basic boid agent that implements core flocking behavior."""
    dx = ally_x[:, None, :] - ally_x[:, :, None]
    dy = ally_y[:, None, :] - ally_y[:, :, None]
    norm = jnp.sqrt(dx ** 2 + dy ** 2)

    x_action = jnp.zeros_like(ally_vx)
    y_action = jnp.zeros_like(ally_vy)

    # Seperation
    separation_mask = jnp.tril(norm < SEPERATION_RADIUS, -1)
    x_action += (dx * separation_mask).sum(axis=1) * SEPERATION_WEIGHT
    y_action += (dy * separation_mask).sum(axis=1) * SEPERATION_WEIGHT

    # Alignment
    alignment_mask = jnp.tril((norm < PERCEPTION_RADIUS) & (norm > SEPERATION_RADIUS), -1)
    vx_total = jnp.vecmat(ally_vx, alignment_mask)
    vy_total = jnp.vecmat(ally_vy, alignment_mask)
    alignment_count = jnp.sum(alignment_mask, axis=1)
    vx_avg = jnp.where(alignment_count > 0, vx_total / alignment_count, ally_vx)
    vy_avg = jnp.where(alignment_count > 0, vy_total / alignment_count, ally_vy)
    x_action += ((vx_avg - ally_vx) * ALIGNMENT_WEIGHT)
    y_action += ((vy_avg - ally_vy) * ALIGNMENT_WEIGHT)

    # Cohesion
    cohesion_mask = jnp.tril(norm < PERCEPTION_RADIUS, -1)
    x_total = jnp.vecmat(ally_x, cohesion_mask)
    y_total = jnp.vecmat(ally_y, cohesion_mask)
    cohesion_count = jnp.sum(cohesion_mask, axis=1)
    x_avg = jnp.where(cohesion_count > 0, x_total / cohesion_count, ally_x)
    y_avg = jnp.where(cohesion_count > 0, y_total / cohesion_count, ally_y)
    x_action += ((x_avg - ally_x) * COHESION_WEIGHT)
    y_action += ((y_avg - ally_y) * COHESION_WEIGHT)
    
    return x_action, y_action
