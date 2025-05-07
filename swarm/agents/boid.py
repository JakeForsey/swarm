import jax
import jax.numpy as jnp

from swarm.env import State


SEPERATION_RADIUS = 0.1
PERCEPTION_RADIUS = 0.3
SEPERATION_WEIGHT = 0.001
ALIGNMENT_WEIGHT = 0.001
COHESION_WEIGHT = 0.01


def act(state: State, team: int, key: jax.random.PRNGKey) -> tuple[jnp.ndarray, jnp.ndarray]:
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
    if team == 1:
        x = state.x1
        y = state.y1
        vx = state.vx1
        vy = state.vy1
    elif team == 2:
        x = state.x2
        y = state.y2
        vx = state.vx2
        vy = state.vy2
    else:
        raise ValueError(f"Invalid team: {team}")
    return _act(x, y, vx, vy)


@jax.jit
def _act(x: jnp.ndarray, y: jnp.ndarray, vx: jnp.ndarray, vy: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    dx = x[:, None, :] - x[:, :, None]
    dy = y[:, None, :] - y[:, :, None]
    norm = jnp.sqrt(dx ** 2 + dy ** 2)

    x_action = jnp.zeros_like(vx)
    y_action = jnp.zeros_like(vy)

    # Seperation
    separation_mask = jnp.tril(norm < SEPERATION_RADIUS, -1)
    x_action += (dx * separation_mask).sum(axis=1) * SEPERATION_WEIGHT
    y_action += (dy * separation_mask).sum(axis=1) * SEPERATION_WEIGHT

    # Alignment
    alignment_mask = jnp.tril((norm < PERCEPTION_RADIUS) & (norm > SEPERATION_RADIUS), -1)
    vx_total = jnp.vecmat(vx, alignment_mask)
    vy_total = jnp.vecmat(vy, alignment_mask)
    alignment_count = jnp.sum(alignment_mask, axis=1)
    vx_avg = jnp.where(alignment_count > 0, vx_total / alignment_count, vx)
    vy_avg = jnp.where(alignment_count > 0, vy_total / alignment_count, vy)
    x_action += ((vx_avg - vx) * ALIGNMENT_WEIGHT)
    y_action += ((vy_avg - vy) * ALIGNMENT_WEIGHT)

    # Cohesion
    cohesion_mask = jnp.tril(norm < PERCEPTION_RADIUS, -1)
    x_total = jnp.vecmat(x, cohesion_mask)
    y_total = jnp.vecmat(y, cohesion_mask)
    cohesion_count = jnp.sum(cohesion_mask, axis=1)
    x_avg = jnp.where(cohesion_count > 0, x_total / cohesion_count, x)
    y_avg = jnp.where(cohesion_count > 0, y_total / cohesion_count, y)
    x_action += ((x_avg - x) * COHESION_WEIGHT)
    y_action += ((y_avg - y) * COHESION_WEIGHT)
    
    return x_action, y_action
