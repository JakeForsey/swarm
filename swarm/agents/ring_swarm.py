import jax
import jax.numpy as jnp

from swarm.env import State


# Formation parameters
FORMATION_CENTER_X = 0.5
FORMATION_CENTER_Y = 0.5
FORMATION_RADIUS = 0.2  # Smaller ring radius
FORMATION_WEIGHT = 0.06  # Position weight
VELOCITY_WEIGHT = 0.08  # Velocity matching weight
DAMPING = 0.1  # Velocity damping factor
RETREAT_HEALTH_THRESHOLD = 0.4
RETREAT_WEIGHT = 0.1  # Retreat weight


def act(state: State, team: int, key: jax.random.PRNGKey) -> tuple[jnp.ndarray, jnp.ndarray]:
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
    if team == 1:
        x = state.x1
        y = state.y1
        vx = state.vx1
        vy = state.vy1
        health = state.health1
    elif team == 2:
        x = state.x2
        y = state.y2
        vx = state.vx2
        vy = state.vy2
        health = state.health2
    else:
        raise ValueError(f"Invalid team: {team}")
    
    return _act(x, y, vx, vy, health)


@jax.jit
def _act(
    x: jnp.ndarray, y: jnp.ndarray,
    vx: jnp.ndarray, vy: jnp.ndarray,
    health: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    # Initialize actions
    x_action = jnp.zeros_like(x)
    y_action = jnp.zeros_like(y)
    
    # Calculate positions relative to formation center
    dx = x - FORMATION_CENTER_X
    dy = y - FORMATION_CENTER_Y
    
    # Handle wrapping by finding shortest path to center
    dx = jnp.where(dx > 0.5, dx - 1.0, dx)
    dx = jnp.where(dx < -0.5, dx + 1.0, dx)
    dy = jnp.where(dy > 0.5, dy - 1.0, dy)
    dy = jnp.where(dy < -0.5, dy + 1.0, dy)
    
    # Calculate target positions on formation circle using agent indices
    num_agents = x.shape[1]
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
    velocity_match_x = target_vx - vx
    velocity_match_y = target_vy - vy
    
    # Calculate retreat movement (towards center)
    retreat_dx = -dx
    retreat_dy = -dy
    
    # Apply movement based on health
    low_health_mask = health < RETREAT_HEALTH_THRESHOLD
    retreat_scale = (RETREAT_HEALTH_THRESHOLD - health) / RETREAT_HEALTH_THRESHOLD
    
    # Combine formation and retreat movements
    x_action += formation_dx * FORMATION_WEIGHT * (1 - low_health_mask)
    y_action += formation_dy * FORMATION_WEIGHT * (1 - low_health_mask)
    x_action += velocity_match_x * VELOCITY_WEIGHT * (1 - low_health_mask)
    y_action += velocity_match_y * VELOCITY_WEIGHT * (1 - low_health_mask)
    x_action += retreat_dx * RETREAT_WEIGHT * low_health_mask * retreat_scale
    y_action += retreat_dy * RETREAT_WEIGHT * low_health_mask * retreat_scale
    
    # Add velocity damping
    x_action -= vx * DAMPING
    y_action -= vy * DAMPING
    
    return x_action, y_action
