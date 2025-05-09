import jax
import jax.numpy as jnp

# Movement parameters
CENTER_WEIGHT = 0.1  # Strong weight for moving to center
DAMPING = 0.15      # Strong damping for stability

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
    # Initialize actions
    x_action = jnp.zeros_like(ally_x)
    y_action = jnp.zeros_like(ally_y)
    
    # Calculate center of gravity
    center_x = jnp.mean(ally_x, axis=1, keepdims=True)
    center_y = jnp.mean(ally_y, axis=1, keepdims=True)
    
    # Calculate relative positions to center
    dx = center_x - ally_x
    dy = center_y - ally_y
    
    # Handle wrapping by finding shortest path to center
    dx = jnp.where(dx > 0.5, dx - 1.0, dx)
    dx = jnp.where(dx < -0.5, dx + 1.0, dx)
    dy = jnp.where(dy > 0.5, dy - 1.0, dy)
    dy = jnp.where(dy < -0.5, dy + 1.0, dy)
    
    # Add center-seeking movement
    x_action += dx * CENTER_WEIGHT
    y_action += dy * CENTER_WEIGHT
    
    # Add velocity damping
    x_action -= ally_vx * DAMPING
    y_action -= ally_vy * DAMPING
    
    return x_action, y_action 