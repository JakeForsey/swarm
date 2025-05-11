import jax
import jax.numpy as jnp

# Formation parameters
FORMATION_CENTER_X = 0.5
FORMATION_CENTER_Y = 0.5
FORMATION_RADIUS = 0.1  # Radius of the vortex
ROTATION_SPEED = 0.2    # Speed of rotation (radians per step)
FORMATION_WEIGHT = 0.06  # Position weight
VELOCITY_WEIGHT = 0.08  # Velocity matching weight
DAMPING = 0.1  # Velocity damping factor

# Combat parameters
CHASE_RADIUS = 0.35
CHASE_WEIGHT = 0.01
MIN_GROUP_SIZE = 2
HEALTH_AGGRESSION_SCALE = 0.8
PERCEPTION_RADIUS = 0.3  # Added perception radius for group size calculation

@jax.jit
def act(
    t,
    key,
    ally_x,
    ally_y,
    ally_vx,
    ally_vy,
    ally_health,
    enemy_y,
    enemy_x,
    enemy_vx,
    enemy_vy,
    enemy_health,
):
    """Vortex swarm agent that rotates around a center point while maintaining formation."""
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
        
    # Calculate target positions on vortex using agent indices
    batch_size, num_agents = ally_x.shape
    agent_indices = jnp.arange(num_agents)

    # Calculate base angles and add time-based rotation
    base_angles = 2 * jnp.pi * agent_indices / num_agents
    rotation_offset = t * ROTATION_SPEED  # Time-based rotation
    target_angles = base_angles[:, None] + rotation_offset
    target_angles = target_angles.T
    
    # Calculate target positions relative to center
    target_dx = FORMATION_RADIUS * jnp.cos(target_angles)
    target_dy = FORMATION_RADIUS * jnp.sin(target_angles)
    
    # Calculate target velocities (tangential to the circle)
    target_vx = -FORMATION_RADIUS * jnp.sin(target_angles) * ROTATION_SPEED
    target_vy = FORMATION_RADIUS * jnp.cos(target_angles) * ROTATION_SPEED
    
    # Calculate formation movement
    formation_dx = target_dx - dx
    formation_dy = target_dy - dy
    
    # Calculate velocity matching
    velocity_match_x = target_vx - ally_vx
    velocity_match_y = target_vy - ally_vy
    
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
    
    # Calculate aggression based on health and group size
    health_aggression = ally_health * HEALTH_AGGRESSION_SCALE
    group_size = jnp.sum(enemy_dist < PERCEPTION_RADIUS, axis=1)
    group_advantage = group_size > MIN_GROUP_SIZE
    
    # Chase if we have group advantage and enemy is within range
    chase_mask = (min_enemy_dist < CHASE_RADIUS) & group_advantage
    chase_strength = chase_mask * health_aggression
    
    # Add combat movement
    x_action += -closest_enemy_dx * chase_strength * CHASE_WEIGHT
    y_action += -closest_enemy_dy * chase_strength * CHASE_WEIGHT
    
    return x_action, y_action
