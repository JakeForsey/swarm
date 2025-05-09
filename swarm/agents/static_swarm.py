import jax
import jax.numpy as jnp

DAMPING = 0.2  # Very strong damping to prevent any movement

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
    # Initialize actions
    x_action = jnp.zeros_like(ally_vx)
    y_action = jnp.zeros_like(ally_vy)
    
    # Apply strong damping to prevent any movement
    x_action -= ally_vx * DAMPING
    y_action -= ally_vy * DAMPING
    
    return x_action, y_action
