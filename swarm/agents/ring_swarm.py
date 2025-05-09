import jax
import jax.numpy as jnp

# Formation parameters
FORMATION_CENTER_X = 0.5
FORMATION_CENTER_Y = 0.5
FORMATION_RADIUS = 0.2  # Smaller ring radius
FORMATION_WEIGHT = 0.06  # Position weight
VELOCITY_WEIGHT = 0.08  # Velocity matching weight
DAMPING = 0.1  # Velocity damping factor
RETREAT_HEALTH_THRESHOLD = 0.4
RETREAT_WEIGHT = 0.1  # Retreat weight

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
    """Ring swarm agent that forms a static ring formation with health-based retreat.
    
    Strategy:
    1. Forms static ring around center (radius 0.15)
    2. Agents with low health (< 0.3) retreat to center
    3. Uses strong formation weight (0.08) for precise positioning
    4. Implements moderate damping (0.1) for stability
    5. No velocity matching or random movement for maximum predictability
    
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
    
    # Calculate target positions on formation circle using agent indices
    num_agents = ally_x.shape[1]
    agent_indices = jnp.arange(num_agents)
    target_angles = 2 * jnp.pi * agent_indices / num_agents
    
    # Calculate target positions relative to center
    target_dx = FORMATION_RADIUS * jnp.cos(target_angles)
    target_dy = FORMATION_RADIUS * jnp.sin(target_angles)
    
    # Calculate target velocities (direction to target)
    target_vx = target_dx - dx
    target_vy = target_dy - dy
    
    # Normalize target velocities
    target_speed = jnp.sqrt(target_vx**2 + target_vy**2)
    target_vx = jnp.where(target_speed > 0, target_vx / target_speed, 0)
    target_vy = jnp.where(target_speed > 0, target_vy / target_speed, 0)
    
    # Calculate formation movement
    formation_dx = target_dx - dx
    formation_dy = target_dy - dy
    
    # Calculate velocity matching
    velocity_match_x = target_vx - ally_vx
    velocity_match_y = target_vy - ally_vy
    
    # Calculate retreat movement (towards center)
    retreat_dx = -dx
    retreat_dy = -dy
    
    # Apply movement based on health
    low_health_mask = ally_health < RETREAT_HEALTH_THRESHOLD
    retreat_scale = (RETREAT_HEALTH_THRESHOLD - ally_health) / RETREAT_HEALTH_THRESHOLD
    
    # Combine formation and retreat movements
    x_action += formation_dx * FORMATION_WEIGHT * (1 - low_health_mask)
    y_action += formation_dy * FORMATION_WEIGHT * (1 - low_health_mask)
    x_action += velocity_match_x * VELOCITY_WEIGHT * (1 - low_health_mask)
    y_action += velocity_match_y * VELOCITY_WEIGHT * (1 - low_health_mask)
    x_action += retreat_dx * RETREAT_WEIGHT * low_health_mask * retreat_scale
    y_action += retreat_dy * RETREAT_WEIGHT * low_health_mask * retreat_scale
    
    # Add velocity damping
    x_action -= ally_vx * DAMPING
    y_action -= ally_vy * DAMPING
    
    return x_action, y_action
