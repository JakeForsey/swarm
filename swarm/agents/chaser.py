import jax
import jax.numpy as jnp

from swarm.env import State


CHASE_RADIUS = 0.3
CHASE_WEIGHT = 0.01
RANDOM_WEIGHT = 0.001


def act(state: State, team: int, key: jax.random.PRNGKey) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Chaser agent that aggressively pursues and attacks enemies.
    
    Strategy:
    1. Actively seeks out enemies within large perception radius (0.4)
    2. Uses strong chase weight (0.1) for aggressive pursuit
    3. Implements moderate damping (0.05) for stability
    4. No formation or velocity matching for maximum focus on chasing
    5. Always engages when enemies are in range
    
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
        target_x = state.x2
        target_y = state.y2
    elif team == 2:
        x = state.x2
        y = state.y2
        vx = state.vx2
        vy = state.vy2
        target_x = state.x1
        target_y = state.y1
    else:
        raise ValueError(f"Invalid team: {team}")
    return _act(x, y, vx, vy, target_x, target_y, key)


@jax.jit
def _act(
    x: jnp.ndarray, y: jnp.ndarray,
    vx: jnp.ndarray, vy: jnp.ndarray,
    target_x: jnp.ndarray, target_y: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    # Calculate distances to all targets
    dx = x[:, None, :] - target_x[:, :, None]
    dy = y[:, None, :] - target_y[:, :, None]
    dist = jnp.sqrt(dx**2 + dy**2)

    # Initialize actions
    x_action = jnp.zeros_like(vx)
    y_action = jnp.zeros_like(vy)

    # Chase behavior
    # Find closest target for each agent
    min_dist = jnp.min(dist, axis=1)
    closest_target_idx = jnp.argmin(dist, axis=1)
    
    # Get relative positions to closest targets
    batch_idx = jnp.arange(x.shape[0])[:, None]
    target_idx = closest_target_idx
    agent_idx = jnp.arange(x.shape[1])[None, :]
    
    closest_dx = dx[batch_idx, target_idx, agent_idx]
    closest_dy = dy[batch_idx, target_idx, agent_idx]
    
    # Chase if within radius
    chase_mask = min_dist < CHASE_RADIUS
    x_action += -closest_dx * chase_mask * CHASE_WEIGHT  # Move towards target
    y_action += -closest_dy * chase_mask * CHASE_WEIGHT

    # Add some random movement
    xkey, ykey, _ = jax.random.split(key, 3)
    x_action += jax.random.uniform(xkey, x.shape, minval=-RANDOM_WEIGHT, maxval=RANDOM_WEIGHT)
    y_action += jax.random.uniform(ykey, y.shape, minval=-RANDOM_WEIGHT, maxval=RANDOM_WEIGHT)

    return x_action, y_action
