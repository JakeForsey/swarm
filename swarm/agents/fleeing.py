import jax
import jax.numpy as jnp

from swarm.env import State


COHESION_RADIUS = 0.2
FLEE_RADIUS = 0.3
FLEE_WEIGHT = 0.01
COHESION_WEIGHT = 0.005
RANDOM_WEIGHT = 0.001


def act(state: State, team: int, key: jax.random.PRNGKey) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Fleeing agent that avoids enemies and maintains distance.
    
    Strategy:
    1. Actively avoids enemies within large perception radius (0.4)
    2. Uses strong flee weight (0.1) for rapid evasion
    3. Implements moderate damping (0.05) for stability
    4. No formation or velocity matching for maximum focus on fleeing
    5. Always flees when enemies are in range
    
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
        enemy_x = state.x2
        enemy_y = state.y2
    elif team == 2:
        x = state.x2
        y = state.y2
        vx = state.vx2
        vy = state.vy2
        enemy_x = state.x1
        enemy_y = state.y1
    else:
        raise ValueError(f"Invalid team: {team}")
    return _act(x, y, vx, vy, enemy_x, enemy_y, key)


@jax.jit
def _act(
    x: jnp.ndarray, y: jnp.ndarray,
    vx: jnp.ndarray, vy: jnp.ndarray,
    enemy_x: jnp.ndarray, enemy_y: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    # Initialize actions
    x_action = jnp.zeros_like(vx)
    y_action = jnp.zeros_like(vy)

    # Calculate distances to enemies
    enemy_dx = x[:, None, :] - enemy_x[:, :, None]
    enemy_dy = y[:, None, :] - enemy_y[:, :, None]
    enemy_dist = jnp.sqrt(enemy_dx**2 + enemy_dy**2)

    # Flee behavior - move away from closest enemy
    min_enemy_dist = jnp.min(enemy_dist, axis=1)
    closest_enemy_idx = jnp.argmin(enemy_dist, axis=1)
    
    batch_idx = jnp.arange(x.shape[0])[:, None]
    enemy_idx = closest_enemy_idx
    agent_idx = jnp.arange(x.shape[1])[None, :]
    
    closest_enemy_dx = enemy_dx[batch_idx, enemy_idx, agent_idx]
    closest_enemy_dy = enemy_dy[batch_idx, enemy_idx, agent_idx]
    
    # Flee if enemy is too close
    flee_mask = min_enemy_dist < FLEE_RADIUS
    x_action += closest_enemy_dx * flee_mask * FLEE_WEIGHT  # Move away from enemy
    y_action += closest_enemy_dy * flee_mask * FLEE_WEIGHT

    # Calculate distances to allies
    ally_dx = x[:, None, :] - x[:, :, None]
    ally_dy = y[:, None, :] - y[:, :, None]
    ally_dist = jnp.sqrt(ally_dx**2 + ally_dy**2)

    # Cohesion behavior - stay close to allies
    # Find average position of nearby allies
    nearby_mask = (ally_dist < COHESION_RADIUS) & (ally_dist > 0)  # Exclude self
    x_total = jnp.sum(x[:, None, :] * nearby_mask, axis=1)
    y_total = jnp.sum(y[:, None, :] * nearby_mask, axis=1)
    nearby_count = jnp.sum(nearby_mask, axis=1)
    
    x_avg = jnp.where(nearby_count > 0, x_total / nearby_count, x)
    y_avg = jnp.where(nearby_count > 0, y_total / nearby_count, y)
    
    # Move towards average position of nearby allies
    x_action += (x_avg - x) * COHESION_WEIGHT
    y_action += (y_avg - y) * COHESION_WEIGHT

    # Add some random movement
    xkey, ykey, _ = jax.random.split(key, 3)
    x_action += jax.random.uniform(xkey, x.shape, minval=-RANDOM_WEIGHT, maxval=RANDOM_WEIGHT)
    y_action += jax.random.uniform(ykey, y.shape, minval=-RANDOM_WEIGHT, maxval=RANDOM_WEIGHT)

    return x_action, y_action
