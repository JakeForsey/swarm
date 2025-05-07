import jax
import jax.numpy as jnp

from swarm.env import State


# Movement parameters
CENTER_WEIGHT = 0.1  # Strong weight for moving to center
DAMPING = 0.15      # Strong damping for stability


def act(state: State, team: int, key: jax.random.PRNGKey) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Center swarm agent that moves all agents to their center of gravity.
    
    Strategy:
    1. Calculates team's center of gravity
    2. Moves all agents towards center with strong weight (0.1)
    3. Uses strong damping (0.15) to maintain position
    4. No combat or enemy awareness
    5. Pure center-seeking behavior
    
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
def _act(
    x: jnp.ndarray, y: jnp.ndarray,
    vx: jnp.ndarray, vy: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    # Initialize actions
    x_action = jnp.zeros_like(x)
    y_action = jnp.zeros_like(y)
    
    # Calculate center of gravity
    center_x = jnp.mean(x, axis=1, keepdims=True)
    center_y = jnp.mean(y, axis=1, keepdims=True)
    
    # Calculate relative positions to center
    dx = center_x - x
    dy = center_y - y
    
    # Handle wrapping by finding shortest path to center
    dx = jnp.where(dx > 0.5, dx - 1.0, dx)
    dx = jnp.where(dx < -0.5, dx + 1.0, dx)
    dy = jnp.where(dy > 0.5, dy - 1.0, dy)
    dy = jnp.where(dy < -0.5, dy + 1.0, dy)
    
    # Add center-seeking movement
    x_action += dx * CENTER_WEIGHT
    y_action += dy * CENTER_WEIGHT
    
    # Add velocity damping
    x_action -= vx * DAMPING
    y_action -= vy * DAMPING
    
    return x_action, y_action 