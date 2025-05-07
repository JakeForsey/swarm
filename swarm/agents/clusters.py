import jax
import jax.numpy as jnp

from swarm.env import State


# Formation parameters
RING_CENTER_X = 0.5  # Fixed center X coordinate
RING_CENTER_Y = 0.5  # Fixed center Y coordinate
RING_RADIUS = 0.3    # Radius of the ring formation
MIN_SPACING = 0.1    # Minimum spacing between agents
RETREAT_HEALTH_THRESHOLD = 0.5  # Health threshold for retreating
RETREAT_SPEED = 0.01  # Speed at which agents retreat to center
FORMATION_WEIGHT = 0.02  # Weight for maintaining formation
RETREAT_WEIGHT = 0.03  # Weight for retreating to center
RANDOM_WEIGHT = 0.001  # Small random movement for exploration


def act(state: State, team: int, key: jax.random.PRNGKey) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Clusters agent that forms multiple independent groups for coordinated movement.
    
    Strategy:
    1. Divides agents into NUM_CLUSTERS (4) independent groups
    2. Each cluster maintains moderate formation (radius 0.15)
    3. Uses balanced velocity matching (0.05) and damping (0.1)
    4. Engages in combat when:
       - Enemy within chase radius (0.25)
       - Group size advantage (2+ agents)
       - Health above threshold (0.3)
    5. Implements moderate perception radius (0.3)
    
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
    
    return _act(x, y, vx, vy, health, key)


@jax.jit
def _act(
    x: jnp.ndarray, y: jnp.ndarray,
    vx: jnp.ndarray, vy: jnp.ndarray,
    health: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    # Initialize actions
    x_action = jnp.zeros_like(vx)
    y_action = jnp.zeros_like(vy)
    
    # Calculate positions relative to fixed center
    dx = x - RING_CENTER_X
    dy = y - RING_CENTER_Y
    
    # Handle wrapping by finding shortest path to center
    dx = jnp.where(dx > 0.5, dx - 1.0, dx)
    dx = jnp.where(dx < -0.5, dx + 1.0, dx)
    dy = jnp.where(dy > 0.5, dy - 1.0, dy)
    dy = jnp.where(dy < -0.5, dy + 1.0, dy)
    
    # Calculate current distance and angle from center
    dist = jnp.sqrt(dx**2 + dy**2)
    angle = jnp.arctan2(dy, dx)
    
    # Calculate target positions on the ring
    num_agents = x.shape[1]
    target_angles = jnp.linspace(0, 2 * jnp.pi, num_agents, endpoint=False)
    target_dx = RING_RADIUS * jnp.cos(target_angles)
    target_dy = RING_RADIUS * jnp.sin(target_angles)
    
    # Calculate formation movement
    formation_dx = target_dx - dx
    formation_dy = target_dy - dy
    x_action += formation_dx * FORMATION_WEIGHT
    y_action += formation_dy * FORMATION_WEIGHT
    
    # Health-based retreat
    low_health_mask = health < RETREAT_HEALTH_THRESHOLD
    retreat_scale = (RETREAT_HEALTH_THRESHOLD - health) / RETREAT_HEALTH_THRESHOLD
    
    # Retreat towards center
    retreat_dx = -dx / (dist + 1e-6) * RETREAT_SPEED
    retreat_dy = -dy / (dist + 1e-6) * RETREAT_SPEED
    
    # Apply retreat only to low health agents
    x_action += retreat_dx * low_health_mask * retreat_scale * RETREAT_WEIGHT
    y_action += retreat_dy * low_health_mask * retreat_scale * RETREAT_WEIGHT
    
    # Maintain minimum spacing between agents
    agent_dx = x[:, None, :] - x[:, :, None]
    agent_dy = y[:, None, :] - y[:, :, None]
    
    # Handle wrapping for inter-agent distances
    agent_dx = jnp.where(agent_dx > 0.5, agent_dx - 1.0, agent_dx)
    agent_dx = jnp.where(agent_dx < -0.5, agent_dx + 1.0, agent_dx)
    agent_dy = jnp.where(agent_dy > 0.5, agent_dy - 1.0, agent_dy)
    agent_dy = jnp.where(agent_dy < -0.5, agent_dy + 1.0, agent_dy)
    
    agent_dist = jnp.sqrt(agent_dx**2 + agent_dy**2)
    
    # Calculate repulsion for agents too close
    too_close = (agent_dist < MIN_SPACING) & (agent_dist > 0)
    repulsion_dx = agent_dx * too_close
    repulsion_dy = agent_dy * too_close
    
    x_action += jnp.sum(repulsion_dx, axis=2) * FORMATION_WEIGHT
    y_action += jnp.sum(repulsion_dy, axis=2) * FORMATION_WEIGHT
    
    # Add small random movement for exploration
    xkey, ykey, _ = jax.random.split(key, 3)
    x_action += jax.random.uniform(xkey, x.shape, minval=-RANDOM_WEIGHT, maxval=RANDOM_WEIGHT)
    y_action += jax.random.uniform(ykey, y.shape, minval=-RANDOM_WEIGHT, maxval=RANDOM_WEIGHT)
    
    return x_action, y_action 