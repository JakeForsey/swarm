import jax
import jax.numpy as jnp

SEPERATION_RADIUS = 0.1
PERCEPTION_RADIUS = 0.3
SEPERATION_WEIGHT = 0.001
ALIGNMENT_WEIGHT = 0.001
COHESION_WEIGHT = 0.01

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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Basic boid agent that implements core flocking behavior.
    
    Strategy:
    1. Maintains moderate separation (radius 0.15) from other boids
    2. Uses balanced alignment (0.05) and cohesion (0.05) weights
    3. Implements moderate damping (0.1) for stability
    4. No combat or enemy awareness
    5. Pure flocking behavior for coordinated movement
    
    Parameters:
        state: Current game state containing positions, velocities, and health
        team: Team identifier (1 or 2)
        key: Random key for any stochastic operations
    
    Returns:
        Tuple of x and y actions for each agent
    """
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
