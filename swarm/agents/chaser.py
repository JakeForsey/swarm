import jax
import jax.numpy as jnp

CHASE_RADIUS = 0.3
CHASE_WEIGHT = 0.01
RANDOM_WEIGHT = 0.001

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
    # Calculate distances to all targets
    dx = ally_x[:, None, :] - enemy_x[:, :, None]
    dy = ally_y[:, None, :] - enemy_y[:, :, None]
    dist = jnp.sqrt(dx**2 + dy**2)

    # Initialize actions
    x_action = jnp.zeros_like(ally_vx)
    y_action = jnp.zeros_like(ally_vy)

    # Chase behavior
    # Find closest target for each agent
    min_dist = jnp.min(dist, axis=1)
    closest_target_idx = jnp.argmin(dist, axis=1)
    
    # Get relative positions to closest targets
    batch_idx = jnp.arange(ally_x.shape[0])[:, None]
    target_idx = closest_target_idx
    agent_idx = jnp.arange(ally_x.shape[1])[None, :]
    
    closest_dx = dx[batch_idx, target_idx, agent_idx]
    closest_dy = dy[batch_idx, target_idx, agent_idx]
    
    # Chase if within radius
    chase_mask = min_dist < CHASE_RADIUS
    x_action += -closest_dx * chase_mask * CHASE_WEIGHT  # Move towards target
    y_action += -closest_dy * chase_mask * CHASE_WEIGHT

    # Add some random movement
    xkey, ykey, _ = jax.random.split(key, 3)
    x_action += jax.random.uniform(xkey, ally_x.shape, minval=-RANDOM_WEIGHT, maxval=RANDOM_WEIGHT)
    y_action += jax.random.uniform(ykey, ally_y.shape, minval=-RANDOM_WEIGHT, maxval=RANDOM_WEIGHT)

    return x_action, y_action
