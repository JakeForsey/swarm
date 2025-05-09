import jax
import jax.numpy as jnp

COHESION_RADIUS = 0.2
FLEE_RADIUS = 0.3
FLEE_WEIGHT = 0.01
COHESION_WEIGHT = 0.005
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
    # Initialize actions
    x_action = jnp.zeros_like(ally_vx)
    y_action = jnp.zeros_like(ally_vy)

    # Calculate distances to enemies
    enemy_dx = ally_x[:, None, :] - enemy_x[:, :, None]
    enemy_dy = ally_y[:, None, :] - enemy_y[:, :, None]
    enemy_dist = jnp.sqrt(enemy_dx**2 + enemy_dy**2)

    # Flee behavior - move away from closest enemy
    min_enemy_dist = jnp.min(enemy_dist, axis=1)
    closest_enemy_idx = jnp.argmin(enemy_dist, axis=1)
    
    batch_idx = jnp.arange(ally_x.shape[0])[:, None]
    enemy_idx = closest_enemy_idx
    agent_idx = jnp.arange(ally_x.shape[1])[None, :]
    
    closest_enemy_dx = enemy_dx[batch_idx, enemy_idx, agent_idx]
    closest_enemy_dy = enemy_dy[batch_idx, enemy_idx, agent_idx]
    
    # Flee if enemy is too close
    flee_mask = min_enemy_dist < FLEE_RADIUS
    x_action += closest_enemy_dx * flee_mask * FLEE_WEIGHT  # Move away from enemy
    y_action += closest_enemy_dy * flee_mask * FLEE_WEIGHT

    # Calculate distances to allies
    ally_dx = ally_x[:, None, :] - ally_x[:, :, None]
    ally_dy = ally_y[:, None, :] - ally_y[:, :, None]
    ally_dist = jnp.sqrt(ally_dx**2 + ally_dy**2)

    # Cohesion behavior - stay close to allies
    # Find average position of nearby allies
    nearby_mask = (ally_dist < COHESION_RADIUS) & (ally_dist > 0)  # Exclude self
    x_total = jnp.sum(ally_x[:, None, :] * nearby_mask, axis=1)
    y_total = jnp.sum(ally_y[:, None, :] * nearby_mask, axis=1)
    nearby_count = jnp.sum(nearby_mask, axis=1)
    
    x_avg = jnp.where(nearby_count > 0, x_total / nearby_count, ally_x)
    y_avg = jnp.where(nearby_count > 0, y_total / nearby_count, ally_y)
    
    # Move towards average position of nearby allies
    x_action += (x_avg - ally_x) * COHESION_WEIGHT
    y_action += (y_avg - ally_y) * COHESION_WEIGHT

    # Add some random movement
    xkey, ykey, _ = jax.random.split(key, 3)
    x_action += jax.random.uniform(xkey, ally_x.shape, minval=-RANDOM_WEIGHT, maxval=RANDOM_WEIGHT)
    y_action += jax.random.uniform(ykey, ally_y.shape, minval=-RANDOM_WEIGHT, maxval=RANDOM_WEIGHT)

    return x_action, y_action
