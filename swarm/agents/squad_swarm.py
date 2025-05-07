import jax
import jax.numpy as jnp

from swarm.env import State


# Formation parameters
NUM_SQUADS = 4  # Number of independent squads
SQUAD_RADIUS = 0.12  # Slightly larger formation
FORMATION_WEIGHT = 0.06  # Reduced formation weight
VELOCITY_WEIGHT = 0.08   # Reduced velocity matching
DAMPING = 0.1          # Reduced damping

# Combat parameters
CHASE_RADIUS = 0.25     # Increased chase radius
CHASE_WEIGHT = 0.03    # Reduced chase weight
MIN_SQUAD_SIZE = 2     # Keep minimum squad size
HEALTH_THRESHOLD = 0.2  # Lower health threshold


def act(state: State, team: int, key: jax.random.PRNGKey) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Squad swarm agent that forms multiple independent squads in fixed positions.
    
    Strategy:
    1. Divides agents into NUM_SQUADS (4) independent squads
    2. Each squad maintains a tight formation (radius 0.12) around a fixed position:
       - Squad 0: Top left corner
       - Squad 1: Top right corner
       - Squad 2: Bottom left corner
       - Squad 3: Bottom right corner
    3. Squads independently engage enemies when:
       - Enemy is within chase radius (0.25)
       - Squad has minimum size (2+ agents)
       - Squad health above threshold (0.2)
    4. Uses velocity matching and damping for smooth movement
    
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
        enemy_x = state.x2
        enemy_y = state.y2
    elif team == 2:
        x = state.x2
        y = state.y2
        vx = state.vx2
        vy = state.vy2
        health = state.health2
        enemy_x = state.x1
        enemy_y = state.y1
    else:
        raise ValueError(f"Invalid team: {team}")
    
    return _act(x, y, vx, vy, health, enemy_x, enemy_y, key)


@jax.jit
def _act(
    x: jnp.ndarray, y: jnp.ndarray,
    vx: jnp.ndarray, vy: jnp.ndarray,
    health: jnp.ndarray,
    enemy_x: jnp.ndarray, enemy_y: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    # Initialize actions
    x_action = jnp.zeros_like(x)
    y_action = jnp.zeros_like(y)
    
    # Calculate squad assignments
    num_agents = x.shape[1]
    agents_per_squad = num_agents // NUM_SQUADS
    squad_assignments = jnp.arange(num_agents) // agents_per_squad
    
    # Initialize squad centers
    squad_centers_x = jnp.zeros((x.shape[0], NUM_SQUADS))
    squad_centers_y = jnp.zeros((x.shape[0], NUM_SQUADS))
    
    # Calculate squad centers
    for squad in range(NUM_SQUADS):
        squad_mask = squad_assignments == squad
        # Calculate mean position for each batch with proper wrapping
        squad_x = x * squad_mask
        squad_y = y * squad_mask
        
        # Handle wrapping by shifting coordinates to be centered around the first agent
        first_agent_x = jnp.where(squad_mask, x, 0.0)[:, 0]  # Remove extra dimension
        first_agent_y = jnp.where(squad_mask, y, 0.0)[:, 0]  # Remove extra dimension
        
        # Shift coordinates to be relative to first agent
        dx = squad_x - first_agent_x[:, None]  # Add dimension for broadcasting
        dy = squad_y - first_agent_y[:, None]  # Add dimension for broadcasting
        
        # Handle wrapping
        dx = jnp.where(dx > 0.5, dx - 1.0, dx)
        dx = jnp.where(dx < -0.5, dx + 1.0, dx)
        dy = jnp.where(dy > 0.5, dy - 1.0, dy)
        dy = jnp.where(dy < -0.5, dy + 1.0, dy)
        
        # Calculate mean relative position
        mean_dx = jnp.mean(dx, axis=1)  # Remove keepdims
        mean_dy = jnp.mean(dy, axis=1)  # Remove keepdims
        
        # Convert back to absolute coordinates
        squad_centers_x = squad_centers_x.at[:, squad].set(
            jnp.mod(first_agent_x + mean_dx, 1.0)
        )
        squad_centers_y = squad_centers_y.at[:, squad].set(
            jnp.mod(first_agent_y + mean_dy, 1.0)
        )
    
    # Calculate positions relative to squad centers
    dx = x - squad_centers_x[:, squad_assignments]
    dy = y - squad_centers_y[:, squad_assignments]
    
    # Handle wrapping by finding shortest path to center
    dx = jnp.where(dx > 0.5, dx - 1.0, dx)
    dx = jnp.where(dx < -0.5, dx + 1.0, dx)
    dy = jnp.where(dy > 0.5, dy - 1.0, dy)
    dy = jnp.where(dy < -0.5, dy + 1.0, dy)
    
    # Calculate target positions on squad circle
    agent_angles = jnp.linspace(0, 2 * jnp.pi, agents_per_squad, endpoint=False)
    target_dx = SQUAD_RADIUS * jnp.cos(agent_angles[squad_assignments % agents_per_squad])
    target_dy = SQUAD_RADIUS * jnp.sin(agent_angles[squad_assignments % agents_per_squad])
    
    # Calculate formation movement
    formation_dx = target_dx - dx
    formation_dy = target_dy - dy
    
    # Calculate velocity matching within squads
    velocity_match_x = jnp.zeros_like(vx)
    velocity_match_y = jnp.zeros_like(vy)
    
    for squad in range(NUM_SQUADS):
        squad_mask = squad_assignments == squad
        squad_vx = vx * squad_mask
        squad_vy = vy * squad_mask
        
        # Calculate squad size and average velocity
        squad_size = jnp.sum(squad_mask)
        vx_avg = jnp.where(squad_size > 0, 
                          jnp.sum(squad_vx) / squad_size, 
                          0.0)
        vy_avg = jnp.where(squad_size > 0, 
                          jnp.sum(squad_vy) / squad_size, 
                          0.0)
        
        # Apply velocity matching only to squad members
        velocity_match_x += (vx_avg - vx) * squad_mask
        velocity_match_y += (vy_avg - vy) * squad_mask
    
    # Add formation and velocity matching forces
    x_action += formation_dx * FORMATION_WEIGHT
    y_action += formation_dy * FORMATION_WEIGHT
    x_action += velocity_match_x * VELOCITY_WEIGHT
    y_action += velocity_match_y * VELOCITY_WEIGHT
    
    # Add velocity damping
    x_action -= vx * DAMPING
    y_action -= vy * DAMPING
    
    # Combat behavior
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
    
    # Calculate squad-based combat decisions
    for squad in range(NUM_SQUADS):
        squad_mask = squad_assignments == squad
        squad_size = jnp.sum(squad_mask)
        squad_health = jnp.mean(health * squad_mask)
        
        # Calculate group advantage within squad
        squad_enemy_dist = enemy_dist * squad_mask[None, :]  # Broadcast squad mask
        squad_group_size = jnp.sum(squad_enemy_dist < CHASE_RADIUS)
        group_advantage = squad_group_size > MIN_SQUAD_SIZE
        health_aggression = squad_health > HEALTH_THRESHOLD
        
        # Chase if squad has advantage and good health
        chase_mask = (min_enemy_dist < CHASE_RADIUS) & group_advantage & health_aggression & squad_mask
        x_action += -closest_enemy_dx * chase_mask * CHASE_WEIGHT
        y_action += -closest_enemy_dy * chase_mask * CHASE_WEIGHT
    
    return x_action, y_action
