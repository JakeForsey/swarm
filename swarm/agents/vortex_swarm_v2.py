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
CHASE_RADIUS = 0.3 # Decreased from 0.35
CHASE_WEIGHT = 0.01
MIN_GROUP_SIZE = 2
HEALTH_AGGRESSION_SCALE = 0.8
PERCEPTION_RADIUS = 0.3  # Added perception radius for group size calculation
RETREAT_HEALTH_THRESHOLD = 0.35 
RETREAT_WEIGHT = 0.1 # Reverted from 0.15

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
    """Vortex swarm agent that rotates around a center point while maintaining formation.
    
    Strategy:
    1. Forms rotating formation around center (radius 0.15)
    2. Rotates at constant speed (0.2 radians per step)
    3. Uses velocity matching (0.08) for smooth rotation
    4. Engages in combat when:
       - Enemy within chase radius (0.25)
       - Group size advantage (2+ agents)
       - Health above threshold (0.3)
    5. Implements moderate damping (0.1) for stability
    
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
    
    # --- Combat and Retreat Logic ---
    
    # Calculate distances to all enemies for targeting and retreat decisions
    enemy_dx = ally_x[:, None, :] - enemy_x[:, :, None]
    enemy_dy = ally_y[:, None, :] - enemy_y[:, :, None]
    
    # Handle wrapping for enemy distances
    enemy_dx = jnp.where(enemy_dx > 0.5, enemy_dx - 1.0, enemy_dx)
    enemy_dx = jnp.where(enemy_dx < -0.5, enemy_dx + 1.0, enemy_dx)
    enemy_dy = jnp.where(enemy_dy > 0.5, enemy_dy - 1.0, enemy_dy)
    enemy_dy = jnp.where(enemy_dy < -0.5, enemy_dy + 1.0, enemy_dy)
    
    enemy_dist = jnp.sqrt(enemy_dx**2 + enemy_dy**2)
    
    # 1. Retreat Logic (Overrides Combat/Formation if triggered)
    should_retreat = ally_health < RETREAT_HEALTH_THRESHOLD
    
    # Find the actual closest enemy for retreat direction calculation
    # Need batch_idx and agent_idx defined first
    batch_idx = jnp.arange(ally_x.shape[0])[:, None]
    agent_idx = jnp.arange(ally_x.shape[1])[None, :]
    closest_enemy_idx_for_retreat = jnp.argmin(enemy_dist, axis=1)
    
    retreat_dx = enemy_dx[batch_idx, closest_enemy_idx_for_retreat, agent_idx]
    retreat_dy = enemy_dy[batch_idx, closest_enemy_idx_for_retreat, agent_idx]
    
    # Normalize retreat direction (move directly away from closest enemy)
    retreat_magnitude = jnp.sqrt(retreat_dx**2 + retreat_dy**2) + 1e-6
    # Move AWAY, so use +retreat_dx/dy
    retreat_move_x = (retreat_dx / retreat_magnitude) * RETREAT_WEIGHT 
    retreat_move_y = (retreat_dy / retreat_magnitude) * RETREAT_WEIGHT
    
    # Apply retreat movement if should_retreat is true, overwriting previous formation/damping forces
    x_action = jnp.where(should_retreat, retreat_move_x, x_action)
    y_action = jnp.where(should_retreat, retreat_move_y, y_action)

    # 2. Combat Logic (Only applies if not retreating)
    # Find lowest health enemy within CHASE_RADIUS
    enemy_health_within_radius = jnp.where(enemy_dist < CHASE_RADIUS, enemy_health[:, :, None], jnp.inf)
    target_enemy_idx = jnp.argmin(enemy_health_within_radius, axis=1) # Shape: (batch_size, num_agents)
    
    # Get distance to the target enemy
    target_enemy_dist = enemy_dist[batch_idx, target_enemy_idx, agent_idx]

    # Get relative positions to target enemies
    target_enemy_dx = enemy_dx[batch_idx, target_enemy_idx, agent_idx]
    target_enemy_dy = enemy_dy[batch_idx, target_enemy_idx, agent_idx]
    
    # Get health of target enemies
    target_enemy_health = enemy_health[batch_idx, target_enemy_idx]

    # Calculate aggression based on health difference and group size
    health_advantage = ally_health - target_enemy_health # Use target enemy health
    health_aggression = jnp.maximum(0, health_advantage) * HEALTH_AGGRESSION_SCALE
    
    # Calculate group size around self (using PERCEPTION_RADIUS, different from CHASE_RADIUS)
    ally_dx = ally_x[:, None, :] - ally_x[:, :, None]
    ally_dy = ally_y[:, None, :] - ally_y[:, :, None]
    ally_dist = jnp.sqrt(ally_dx**2 + ally_dy**2)
    group_size = jnp.sum(ally_dist < PERCEPTION_RADIUS, axis=1) # Count allies within perception radius
    group_advantage = group_size > MIN_GROUP_SIZE
    
    # Chase if target enemy is within CHASE_RADIUS, we have group advantage, and health advantage
    can_chase = (target_enemy_dist < CHASE_RADIUS) & group_advantage & (health_advantage > 0)
    chase_strength = can_chase * health_aggression
    
    # Calculate combat movement towards target enemy
    combat_move_x = -target_enemy_dx * chase_strength * CHASE_WEIGHT
    combat_move_y = -target_enemy_dy * chase_strength * CHASE_WEIGHT
    
    # Apply combat movement only if not retreating
    x_action = jnp.where(should_retreat, x_action, x_action + combat_move_x)
    y_action = jnp.where(should_retreat, y_action, y_action + combat_move_y)

    return x_action, y_action
