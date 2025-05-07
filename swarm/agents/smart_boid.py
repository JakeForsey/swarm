import jax
import jax.numpy as jnp

from swarm.env import State


# Core boid parameters
SEPARATION_RADIUS = 0.08  # Slightly looser separation
PERCEPTION_RADIUS = 0.2   # Larger perception radius
SEPARATION_WEIGHT = 0.05  # Stronger separation
ALIGNMENT_WEIGHT = 0.05   # Stronger alignment
COHESION_WEIGHT = 0.06    # Stronger cohesion
DAMPING = 0.1            # Velocity damping

# Combat parameters
CHASE_RADIUS = 0.25      # Larger chase radius
CHASE_WEIGHT = 0.08      # Stronger chase
FLEE_RADIUS = 0.15       # Larger flee radius
FLEE_WEIGHT = 0.06       # Stronger flee
MIN_GROUP_SIZE = 2       # Keep group size requirement
HEALTH_THRESHOLD = 0.3   # Health threshold for aggression


def act(state: State, team: int, key: jax.random.PRNGKey) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Smart boid agent that adapts its behavior based on health and group size.
    
    Strategy:
    1. Maintains moderate formation (radius 0.2) with other boids
    2. Uses balanced velocity matching (0.05) and damping (0.1)
    3. Adapts combat behavior based on conditions:
       - Aggressive when health > 0.3 and group size > 2
       - Defensive when health < 0.3 or alone
       - Neutral otherwise
    4. Implements moderate chase radius (0.25) for engagement
    5. Uses perception radius (0.2) for group awareness
    
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
        health = state.health1
    elif team == 2:
        x = state.x2
        y = state.y2
        vx = state.vx2
        vy = state.vy2
        enemy_x = state.x1
        enemy_y = state.y1
        health = state.health2
    else:
        raise ValueError(f"Invalid team: {team}")
    
    return _act(x, y, vx, vy, enemy_x, enemy_y, health, key)


@jax.jit
def _act(
    x: jnp.ndarray, y: jnp.ndarray,
    vx: jnp.ndarray, vy: jnp.ndarray,
    enemy_x: jnp.ndarray, enemy_y: jnp.ndarray,
    health: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    # Initialize actions
    x_action = jnp.zeros_like(x)
    y_action = jnp.zeros_like(y)
    
    # Add velocity damping
    x_action -= vx * DAMPING
    y_action -= vy * DAMPING
    
    # Calculate distances to allies
    ally_dx = x[:, None, :] - x[:, :, None]
    ally_dy = y[:, None, :] - y[:, :, None]
    
    # Handle wrapping for inter-agent distances
    ally_dx = jnp.where(ally_dx > 0.5, ally_dx - 1.0, ally_dx)
    ally_dx = jnp.where(ally_dx < -0.5, ally_dx + 1.0, ally_dx)
    ally_dy = jnp.where(ally_dy > 0.5, ally_dy - 1.0, ally_dy)
    ally_dy = jnp.where(ally_dy < -0.5, ally_dy + 1.0, ally_dy)
    
    ally_dist = jnp.sqrt(ally_dx**2 + ally_dy**2)
    
    # 1. Separation from allies
    separation_mask = jnp.tril(ally_dist < SEPARATION_RADIUS, -1)
    x_action += (ally_dx * separation_mask).sum(axis=1) * SEPARATION_WEIGHT
    y_action += (ally_dy * separation_mask).sum(axis=1) * SEPARATION_WEIGHT
    
    # 2. Alignment with nearby allies
    alignment_mask = jnp.tril((ally_dist < PERCEPTION_RADIUS) & (ally_dist > SEPARATION_RADIUS), -1)
    vx_total = jnp.sum(vx[:, None, :] * alignment_mask, axis=1)
    vy_total = jnp.sum(vy[:, None, :] * alignment_mask, axis=1)
    alignment_count = jnp.sum(alignment_mask, axis=1)
    vx_avg = jnp.where(alignment_count > 0, vx_total / alignment_count, vx)
    vy_avg = jnp.where(alignment_count > 0, vy_total / alignment_count, vy)
    x_action += (vx_avg - vx) * ALIGNMENT_WEIGHT
    y_action += (vy_avg - vy) * ALIGNMENT_WEIGHT
    
    # 3. Cohesion with nearby allies
    cohesion_mask = jnp.tril(ally_dist < PERCEPTION_RADIUS, -1)
    x_total = jnp.sum(x[:, None, :] * cohesion_mask, axis=1)
    y_total = jnp.sum(y[:, None, :] * cohesion_mask, axis=1)
    cohesion_count = jnp.sum(cohesion_mask, axis=1)
    x_avg = jnp.where(cohesion_count > 0, x_total / cohesion_count, x)
    y_avg = jnp.where(cohesion_count > 0, y_total / cohesion_count, y)
    x_action += (x_avg - x) * COHESION_WEIGHT
    y_action += (y_avg - y) * COHESION_WEIGHT
    
    # 4. Combat behavior
    # Calculate distances to enemies
    enemy_dx = x[:, None, :] - enemy_x[:, :, None]
    enemy_dy = y[:, None, :] - enemy_y[:, :, None]
    
    # Handle wrapping for enemy distances
    enemy_dx = jnp.where(enemy_dx > 0.5, enemy_dx - 1.0, enemy_dx)
    enemy_dx = jnp.where(enemy_dx < -0.5, enemy_dx + 1.0, enemy_dx)
    enemy_dy = jnp.where(enemy_dy > 0.5, enemy_dy - 1.0, enemy_dy)
    enemy_dy = jnp.where(enemy_dy < -0.5, enemy_dy + 1.0, enemy_dy)
    
    enemy_dist = jnp.sqrt(enemy_dx**2 + enemy_dy**2)
    
    # Find closest enemy for each agent
    min_enemy_dist = jnp.min(enemy_dist, axis=1)
    closest_enemy_idx = jnp.argmin(enemy_dist, axis=1)
    
    # Get relative positions to closest enemies
    batch_idx = jnp.arange(x.shape[0])[:, None]
    enemy_idx = closest_enemy_idx
    agent_idx = jnp.arange(x.shape[1])[None, :]
    
    closest_enemy_dx = enemy_dx[batch_idx, enemy_idx, agent_idx]
    closest_enemy_dy = enemy_dy[batch_idx, enemy_idx, agent_idx]
    
    # Calculate group size advantage and health-based aggression
    group_size = jnp.sum(ally_dist < PERCEPTION_RADIUS, axis=1)
    group_advantage = group_size > MIN_GROUP_SIZE
    health_aggression = health > HEALTH_THRESHOLD
    
    # Chase if we have group advantage and good health
    chase_mask = (min_enemy_dist < CHASE_RADIUS) & group_advantage & health_aggression
    x_action += -closest_enemy_dx * chase_mask * CHASE_WEIGHT
    y_action += -closest_enemy_dy * chase_mask * CHASE_WEIGHT
    
    # Flee if we're outnumbered or low health and enemy is too close
    flee_mask = (min_enemy_dist < FLEE_RADIUS) & (~group_advantage | ~health_aggression)
    x_action += closest_enemy_dx * flee_mask * FLEE_WEIGHT
    y_action += closest_enemy_dy * flee_mask * FLEE_WEIGHT
    
    return x_action, y_action
