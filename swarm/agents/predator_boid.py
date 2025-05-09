import jax
import jax.numpy as jnp

# Formation parameters
FORMATION_CENTER_X = 0.5
FORMATION_CENTER_Y = 0.5
FORMATION_RADIUS = 0.12  # Tight formation
ROTATION_SPEED = 0.2     # Moderate rotation
FORMATION_WEIGHT = 0.08  # Strong formation weight
VELOCITY_WEIGHT = 0.1    # Strong velocity matching
DAMPING = 0.15          # Strong damping

# Combat parameters
CHASE_RADIUS = 0.25     # Short chase radius
CHASE_WEIGHT = 0.04     # Strong chase
MIN_GROUP_SIZE = 2      # Group size requirement
HEALTH_THRESHOLD = 0.3  # Health threshold for aggression

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
    """Predator boid agent that aggressively chases and attacks enemies.
    
    Strategy:
    1. Maintains loose formation with other predators (radius 0.2)
    2. Actively seeks out and chases enemies within perception radius (0.3)
    3. Uses strong velocity matching (0.1) for coordinated movement
    4. Implements moderate damping (0.05) for stability
    5. Prioritizes chasing over formation when enemies are nearby
    
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
    
    # Calculate positions relative to formation center
    dx = ally_x - FORMATION_CENTER_X
    dy = ally_y - FORMATION_CENTER_Y
    
    # Handle wrapping by finding shortest path to center
    dx = jnp.where(dx > 0.5, dx - 1.0, dx)
    dx = jnp.where(dx < -0.5, dx + 1.0, dx)
    dy = jnp.where(dy > 0.5, dy - 1.0, dy)
    dy = jnp.where(dy < -0.5, dy + 1.0, dy)
    
    # Calculate target positions on formation circle with rotation
    num_agents = ally_x.shape[1]
    base_angles = jnp.linspace(0, 2 * jnp.pi, num_agents, endpoint=False)
    rotation_offset = (t * ROTATION_SPEED)[:, None]  # Shape: (batch_size, 1)
    target_angles = base_angles + rotation_offset  # Now broadcasts to (batch_size, num_agents)
    target_dx = FORMATION_RADIUS * jnp.cos(target_angles)
    target_dy = FORMATION_RADIUS * jnp.sin(target_angles)
    
    # Calculate formation movement
    formation_dx = target_dx - dx
    formation_dy = target_dy - dy
    
    # Calculate velocity matching
    velocity_match_x = -ally_vx  # Damp current velocity
    velocity_match_y = -ally_vy
    
    # Add formation and velocity matching forces
    x_action += formation_dx * FORMATION_WEIGHT
    y_action += formation_dy * FORMATION_WEIGHT
    x_action += velocity_match_x * VELOCITY_WEIGHT
    y_action += velocity_match_y * VELOCITY_WEIGHT
    
    # Add velocity damping
    x_action -= ally_vx * DAMPING
    y_action -= ally_vy * DAMPING
    
    # Combat behavior
    # Calculate distances to enemies
    enemy_dx = ally_x[:, None, :] - enemy_x[:, :, None]
    enemy_dy = ally_y[:, None, :] - enemy_y[:, :, None]
    
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
    batch_idx = jnp.arange(ally_x.shape[0])[:, None]
    enemy_idx = closest_enemy_idx
    agent_idx = jnp.arange(ally_x.shape[1])[None, :]
    
    closest_enemy_dx = enemy_dx[batch_idx, enemy_idx, agent_idx]
    closest_enemy_dy = enemy_dy[batch_idx, enemy_idx, agent_idx]
    
    # Calculate group size and health-based aggression
    group_size = jnp.sum(enemy_dist < CHASE_RADIUS, axis=1)
    group_advantage = group_size > MIN_GROUP_SIZE
    health_aggression = ally_health > HEALTH_THRESHOLD
    
    # Chase if we have group advantage and good health
    chase_mask = (min_enemy_dist < CHASE_RADIUS) & group_advantage & health_aggression
    
    # Add combat movement
    x_action += -closest_enemy_dx * chase_mask * CHASE_WEIGHT
    y_action += -closest_enemy_dy * chase_mask * CHASE_WEIGHT
    
    return x_action, y_action
