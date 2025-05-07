import jax
import jax.numpy as jnp

from swarm.env import State


# Movement parameters
DAMPING = 0.2  # Very strong damping to prevent any movement


def act(state: State, team: int, key: jax.random.PRNGKey) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Static swarm agent that keeps all agents exactly where they start.
    
    Strategy:
    1. Applies strong damping (0.2) to prevent any movement
    2. No formation, combat, or coordinated behavior
    3. Agents stay exactly at their initial positions
    4. Pure static behavior for baseline comparison
    
    Parameters:
        state: Current game state containing positions, velocities, and health
        team: Team identifier (1 or 2)
        key: Random key for any stochastic operations
    
    Returns:
        Tuple of x and y actions for each agent
    """
    if team == 1:
        vx = state.vx1
        vy = state.vy1
    elif team == 2:
        vx = state.vx2
        vy = state.vy2
    else:
        raise ValueError(f"Invalid team: {team}")
    
    return _act(vx, vy)


@jax.jit
def _act(
    vx: jnp.ndarray, vy: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    # Initialize actions
    x_action = jnp.zeros_like(vx)
    y_action = jnp.zeros_like(vy)
    
    # Apply strong damping to prevent any movement
    x_action -= vx * DAMPING
    y_action -= vy * DAMPING
    
    return x_action, y_action 