import jax
import jax.numpy as jnp

from swarm.env import State


# Formation parameters
SQUAD_RADIUS = 0.15      # Slightly looser squad radius
FORMATION_WEIGHT = 0.08  # Stronger formation weight
VELOCITY_WEIGHT = 0.06   # Reduced velocity weight
DAMPING = 0.15          # Stronger damping

# Combat parameters
CHASE_RADIUS = 0.2      # Reduced chase radius
CHASE_WEIGHT = 0.05     # Balanced chase weight
MIN_SQUAD_SIZE = 2      # Keep minimum squad size
HEALTH_THRESHOLD = 0.3  # Increased health threshold


def act(state: State, team: int, key: jax.random.PRNGKey) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Squad swarm that forms multiple tight clusters to chase enemies while maintaining formation.
    
    Strategy:
    1. Forms multiple independent squads of 2+ agents
    2. Each squad maintains tight formation (radius 0.15)
    3. Squads coordinate to chase nearest enemy when healthy
    4. Uses strong damping (0.15) for stable movement
    5. Implements moderate chase radius (0.2) for focused combat
    
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
    
    # Form squads based on proximity
    squad_mask = ally_dist < SQUAD_RADIUS
    
    # Calculate squad centers using relative positions
    # Use the first agent in each squad as reference point
    squad_centers_x = jnp.zeros_like(x)
    squad_centers_y = jnp.zeros_like(y)
    
    for i in range(x.shape[1]):
        # Get agents in this squad
        squad_members = squad_mask[:, i, :]
        
        # Calculate relative positions to first agent
        rel_x = x - x[:, i:i+1]
        rel_y = y - y[:, i:i+1]
        
        # Handle wrapping for relative positions
        rel_x = jnp.where(rel_x > 0.5, rel_x - 1.0, rel_x)
        rel_x = jnp.where(rel_x < -0.5, rel_x + 1.0, rel_x)
        rel_y = jnp.where(rel_y > 0.5, rel_y - 1.0, rel_y)
        rel_y = jnp.where(rel_y < -0.5, rel_y + 1.0, rel_y)
        
        # Calculate mean position relative to first agent
        mean_rel_x = jnp.sum(rel_x * squad_members, axis=1, keepdims=True) / jnp.maximum(jnp.sum(squad_members, axis=1, keepdims=True), 1)
        mean_rel_y = jnp.sum(rel_y * squad_members, axis=1, keepdims=True) / jnp.maximum(jnp.sum(squad_members, axis=1, keepdims=True), 1)
        
        # Convert back to absolute positions
        squad_centers_x = jnp.where(squad_members, x[:, i:i+1] + mean_rel_x, squad_centers_x)
        squad_centers_y = jnp.where(squad_members, y[:, i:i+1] + mean_rel_y, squad_centers_y)
    
    # Move towards squad centers
    dx = squad_centers_x - x
    dy = squad_centers_y - y
    
    # Handle wrapping for movement
    dx = jnp.where(dx > 0.5, dx - 1.0, dx)
    dx = jnp.where(dx < -0.5, dx + 1.0, dx)
    dy = jnp.where(dy > 0.5, dy - 1.0, dy)
    dy = jnp.where(dy < -0.5, dy + 1.0, dy)
    
    x_action += dx * FORMATION_WEIGHT
    y_action += dy * FORMATION_WEIGHT
    
    # Velocity matching within squads
    # Reshape velocities for broadcasting
    vx_expanded = vx[:, None, :]  # Shape: [batch, 1, agents]
    vy_expanded = vy[:, None, :]  # Shape: [batch, 1, agents]
    
    # Calculate average velocity for each squad
    squad_vx = jnp.sum(vx_expanded * squad_mask, axis=2) / jnp.maximum(jnp.sum(squad_mask, axis=2), 1)
    squad_vy = jnp.sum(vy_expanded * squad_mask, axis=2) / jnp.maximum(jnp.sum(squad_mask, axis=2), 1)
    
    x_action += (squad_vx - vx) * VELOCITY_WEIGHT
    y_action += (squad_vy - vy) * VELOCITY_WEIGHT
    
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
    
    # Find closest enemy for each squad
    min_enemy_dist = jnp.min(enemy_dist, axis=1)
    closest_enemy_idx = jnp.argmin(enemy_dist, axis=1)
    
    # Get relative positions to closest enemies
    batch_idx = jnp.arange(x.shape[0])[:, None]
    enemy_idx = closest_enemy_idx
    agent_idx = jnp.arange(x.shape[1])[None, :]
    
    closest_enemy_dx = enemy_dx[batch_idx, enemy_idx, agent_idx]
    closest_enemy_dy = enemy_dy[batch_idx, enemy_idx, agent_idx]
    
    # Calculate squad health and size
    # Reshape health for broadcasting
    health_expanded = health[:, None, :]  # Shape: [batch, 1, agents]
    
    # Calculate average health for each squad
    squad_health = jnp.sum(health_expanded * squad_mask, axis=2) / jnp.maximum(jnp.sum(squad_mask, axis=2), 1)
    squad_size = jnp.sum(squad_mask, axis=2)
    
    # Chase if squad is healthy and has enough members
    chase_mask = (min_enemy_dist < CHASE_RADIUS) & (squad_health > HEALTH_THRESHOLD) & (squad_size >= MIN_SQUAD_SIZE)
    x_action += -closest_enemy_dx * chase_mask * CHASE_WEIGHT
    y_action += -closest_enemy_dy * chase_mask * CHASE_WEIGHT
    
    return x_action, y_action
