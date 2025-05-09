import jax
import jax.numpy as jnp

# Formation parameters
LINE_SPACING = 0.01  # Distance between consecutive agents along the line
FORMATION_WEIGHT = 0.1  # Weight for maintaining line formation
VELOCITY_WEIGHT = 0.08  # Weight for velocity matching
DAMPING = 0.1  # Damping for stability

# Combat parameters
CHASE_RADIUS = 0.25  # Radius to detect enemies
CHASE_WEIGHT = 0.06  # Weight for chasing enemies

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
    """Train swarm agent that forms a line from first alive agent to nearest enemy.
    
    Strategy:
    1. Find first alive agent and nearest enemy
    2. Calculate vector from first agent to nearest enemy
    3. Position each agent along this vector at proportional distances
    4. Use formation weight (0.1) to maintain line formation
    5. Match velocities (0.08) for smooth movement
    6. Chase enemies within radius (0.25) with weight (0.06)
    
    Parameters:
        state: Current game state containing positions, velocities, and health
        team: Team identifier (1 or 2)
        key: Random key for any stochastic operations
    
    Returns:
        Tuple of x and y actions for each agent
    """
    batch_size = ally_x.shape[0]
    num_agents = ally_x.shape[1]

    # Initialize actions
    x_action = jnp.zeros_like(ally_x)
    y_action = jnp.zeros_like(ally_y)
    
    # Find first alive agent
    alive = ally_health > 0
    first_alive_idx = jnp.argmax(alive, axis=1, keepdims=True)
    assert first_alive_idx.shape == (batch_size, 1), f"first_alive_idx.shape: {first_alive_idx.shape}, expected ({batch_size}, 1)"
    
    first_alive_x = ally_x[jnp.arange(batch_size), first_alive_idx[:, 0]]
    first_alive_y = ally_y[jnp.arange(batch_size), first_alive_idx[:, 0]]
    assert first_alive_x.shape == (batch_size, ), f"first_alive_x.shape: {first_alive_x.shape}, expected ({batch_size}, )"
    assert first_alive_y.shape == (batch_size, ), f"first_alive_y.shape: {first_alive_y.shape}, expected ({batch_size}, )"
    
    # Calculate distances to all enemies from first alive agent
    dx_enemy = enemy_x - first_alive_x[:, None]
    dy_enemy = enemy_y - first_alive_y[:, None]
    assert dx_enemy.shape == (batch_size, num_agents)
    assert dy_enemy.shape == (batch_size, num_agents)
    
    enemy_dist = jnp.sqrt(dx_enemy ** 2 + dy_enemy ** 2)
    assert enemy_dist.shape == (batch_size, num_agents)
    
    # Find nearest enemy
    nearest_enemy_idx = jnp.argmin(enemy_dist, axis=1)
    assert nearest_enemy_idx.shape == (batch_size,)
    
    nearest_enemy_x = enemy_x[jnp.arange(batch_size), nearest_enemy_idx]
    nearest_enemy_y = enemy_y[jnp.arange(batch_size), nearest_enemy_idx]
    assert nearest_enemy_x.shape == (batch_size, ), f"nearest_enemy_x.shape: {nearest_enemy_x.shape}, expected ({batch_size}, )"
    assert nearest_enemy_y.shape == (batch_size, ), f"nearest_enemy_y.shape: {nearest_enemy_y.shape}, expected ({batch_size}, )"
    
    # Calculate vector from first alive agent to nearest enemy
    line_dx = nearest_enemy_x - first_alive_x
    line_dy = nearest_enemy_y - first_alive_y
    assert line_dx.shape == (batch_size, ), f"line_dx.shape: {line_dx.shape}, expected ({batch_size}, )"
    assert line_dy.shape == (batch_size, ), f"line_dy.shape: {line_dy.shape}, expected ({batch_size}, )"
    
    # Normalize line direction
    line_length = jnp.sqrt(line_dx ** 2 + line_dy ** 2)
    line_dx = line_dx / (line_length + 1e-6)
    line_dy = line_dy / (line_length + 1e-6)
    
    # Calculate target positions along the line
    # Each agent's target is a distance proportional to its index from the first alive agent
    agent_indices = jnp.repeat(jnp.arange(num_agents)[None, :], batch_size, axis=0)
    assert agent_indices.shape == (batch_size, num_agents)
    target_distances = agent_indices * LINE_SPACING
    assert target_distances.shape == (batch_size, num_agents)
    
    # Calculate target positions
    target_x = first_alive_x[:, None] + line_dx[:, None] * target_distances
    target_y = first_alive_y[:, None] + line_dy[:, None] * target_distances
    assert target_x.shape == (batch_size, num_agents)
    assert target_y.shape == (batch_size, num_agents)
    
    # Handle wrapping for target positions
    target_x = jnp.mod(target_x, 1.0)
    target_y = jnp.mod(target_y, 1.0)
    
    # Calculate movement to target positions
    dx = target_x - ally_x
    dy = target_y - ally_y
        
    # Add formation movement
    x_action += dx * FORMATION_WEIGHT
    y_action += dy * FORMATION_WEIGHT
    
    # # Add chase movement if enemy is within radius
    x_action += line_dx[:, None] * CHASE_WEIGHT
    y_action += line_dy[:, None] * CHASE_WEIGHT
    
    # Calculate average velocity of the swarm
    avg_vx = jnp.mean(ally_vx, axis=1, keepdims=True)
    avg_vy = jnp.mean(ally_vy, axis=1, keepdims=True)
    
    # Add velocity matching
    x_action += (avg_vx - ally_vx) * VELOCITY_WEIGHT
    y_action += (avg_vy - ally_vy) * VELOCITY_WEIGHT
    
    # Add velocity damping
    x_action -= ally_vx * DAMPING
    y_action -= ally_vy * DAMPING
    
    return x_action, y_action
