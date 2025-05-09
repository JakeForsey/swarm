import jax
import jax.numpy as jnp

# Behavior parameters
FORMATION_SCALE = 0.7  # How tightly agents cluster (0 = spread, 1 = tight)
FORMATION_SHAPE = 0.3  # Shape of formation (0 = circle, 1 = line)
AGGRESSIVENESS = 0.6  # How aggressively they approach enemies (0 = defensive, 1 = aggressive)
SMOOTHNESS = 0.8  # How smoothly they change direction (0 = jerky, 1 = smooth)
ATTACK_THRESHOLD = 0.3  # Health threshold to start attacking
RETREAT_THRESHOLD = 0.2  # Health threshold to start retreating

# Movement parameters
MAX_SPEED = 0.01
DAMPING = 0.1

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
    """Parametric swarm agent with tunable behavior.
    
    Strategy:
    1. Maintains formation based on formation_scale and formation_shape
    2. Approaches or retreats from enemies based on health and aggressiveness
    3. Smoothly transitions between behaviors based on smoothness
    
    Parameters:
        state: Current game state containing positions, velocities, and health
        team: Team identifier (1 or 2)
        key: Random key for any stochastic operations
    
    Returns:
        Tuple of x and y actions for each agent
    """
    batch_size, num_agents = ally_x.shape
    
    # Shape assertions for input arrays
    assert ally_x.shape == (batch_size, num_agents), f"ally_x shape {ally_x.shape} != {(batch_size, num_agents)}"
    assert ally_y.shape == (batch_size, num_agents), f"ally_y shape {ally_y.shape} != {(batch_size, num_agents)}"
    assert ally_vx.shape == (batch_size, num_agents), f"ally_vx shape {ally_vx.shape} != {(batch_size, num_agents)}"
    assert ally_vy.shape == (batch_size, num_agents), f"ally_vy shape {ally_vy.shape} != {(batch_size, num_agents)}"
    assert ally_health.shape == (batch_size, num_agents), f"ally_health shape {ally_health.shape} != {(batch_size, num_agents)}"
    assert enemy_x.shape == (batch_size, num_agents), f"enemy_x shape {enemy_x.shape} != {(batch_size, num_agents)}"
    assert enemy_y.shape == (batch_size, num_agents), f"enemy_y shape {enemy_y.shape} != {(batch_size, num_agents)}"
    assert enemy_vx.shape == (batch_size, num_agents), f"enemy_vx shape {enemy_vx.shape} != {(batch_size, num_agents)}"
    assert enemy_vy.shape == (batch_size, num_agents), f"enemy_vy shape {enemy_vy.shape} != {(batch_size, num_agents)}"
    assert enemy_health.shape == (batch_size, num_agents), f"enemy_health shape {enemy_health.shape} != {(batch_size, num_agents)}"
    
    # Compute center of mass and average health
    ally_com_x = jnp.mean(ally_x, axis=1, keepdims=True)
    ally_com_y = jnp.mean(ally_y, axis=1, keepdims=True)
    avg_health = jnp.mean(ally_health, axis=1, keepdims=True)
    
    # Shape assertions for center of mass
    assert ally_com_x.shape == (batch_size, 1), f"ally_com_x shape {ally_com_x.shape} != {(batch_size, 1)}"
    assert ally_com_y.shape == (batch_size, 1), f"ally_com_y shape {ally_com_y.shape} != {(batch_size, 1)}"
    assert avg_health.shape == (batch_size, 1), f"avg_health shape {avg_health.shape} != {(batch_size, 1)}"
    
    # Compute relative positions to center of mass
    rel_x = ally_x - ally_com_x
    rel_y = ally_y - ally_com_y
    
    # Shape assertions for relative positions
    assert rel_x.shape == (batch_size, num_agents), f"rel_x shape {rel_x.shape} != {(batch_size, num_agents)}"
    assert rel_y.shape == (batch_size, num_agents), f"rel_y shape {rel_y.shape} != {(batch_size, num_agents)}"
    
    # Compute formation target positions
    # Circular formation component
    angles = jnp.linspace(0, 2 * jnp.pi, num_agents, endpoint=False)
    circle_x = jnp.cos(angles) * FORMATION_SCALE
    circle_y = jnp.sin(angles) * FORMATION_SCALE
    
    # Linear formation component
    line_x = jnp.linspace(-FORMATION_SCALE, FORMATION_SCALE, num_agents)
    line_y = jnp.zeros_like(line_x)
    
    # Shape assertions for formation components
    assert circle_x.shape == (num_agents,), f"circle_x shape {circle_x.shape} != {(num_agents,)}"
    assert circle_y.shape == (num_agents,), f"circle_y shape {circle_y.shape} != {(num_agents,)}"
    assert line_x.shape == (num_agents,), f"line_x shape {line_x.shape} != {(num_agents,)}"
    assert line_y.shape == (num_agents,), f"line_y shape {line_y.shape} != {(num_agents,)}"
    
    # Blend between circle and line based on formation_shape
    target_x = (1 - FORMATION_SHAPE) * circle_x + FORMATION_SHAPE * line_x
    target_y = (1 - FORMATION_SHAPE) * circle_y + FORMATION_SHAPE * line_y
    
    # Shape assertions for target positions
    assert target_x.shape == (num_agents,), f"target_x shape {target_x.shape} != {(num_agents,)}"
    assert target_y.shape == (num_agents,), f"target_y shape {target_y.shape} != {(num_agents,)}"
    
    # Compute formation forces
    formation_dx = target_x - rel_x
    formation_dy = target_y - rel_y
    
    # Shape assertions for formation forces
    assert formation_dx.shape == (batch_size, num_agents), f"formation_dx shape {formation_dx.shape} != {(batch_size, num_agents)}"
    assert formation_dy.shape == (batch_size, num_agents), f"formation_dy shape {formation_dy.shape} != {(batch_size, num_agents)}"
    
    # Find nearest enemy for each agent
    dx = enemy_x[:, None, :] - ally_x[:, :, None]  # [batch, ally, enemy]
    dy = enemy_y[:, None, :] - ally_y[:, :, None]
    distances = jnp.sqrt(dx**2 + dy**2)
    nearest_enemy_idx = jnp.argmin(distances, axis=-1)  # [batch, ally]
    
    # Shape assertions for enemy distances
    assert dx.shape == (batch_size, num_agents, num_agents), f"dx shape {dx.shape} != {(batch_size, num_agents, num_agents)}"
    assert dy.shape == (batch_size, num_agents, num_agents), f"dy shape {dy.shape} != {(batch_size, num_agents, num_agents)}"
    assert distances.shape == (batch_size, num_agents, num_agents), f"distances shape {distances.shape} != {(batch_size, num_agents, num_agents)}"
    assert nearest_enemy_idx.shape == (batch_size, num_agents), f"nearest_enemy_idx shape {nearest_enemy_idx.shape} != {(batch_size, num_agents)}"
    
    # Get nearest enemy positions using gather
    batch_indices = jnp.arange(batch_size)[:, None]  # [batch, 1]
    nearest_enemy_x = enemy_x[batch_indices, nearest_enemy_idx]  # [batch, ally]
    nearest_enemy_y = enemy_y[batch_indices, nearest_enemy_idx]  # [batch, ally]
    
    # Shape assertions for nearest enemies
    assert batch_indices.shape == (batch_size, 1), f"batch_indices shape {batch_indices.shape} != {(batch_size, 1)}"
    assert nearest_enemy_x.shape == (batch_size, num_agents), f"nearest_enemy_x shape {nearest_enemy_x.shape} != {(batch_size, num_agents)}"
    assert nearest_enemy_y.shape == (batch_size, num_agents), f"nearest_enemy_y shape {nearest_enemy_y.shape} != {(batch_size, num_agents)}"
    
    # Compute enemy forces based on health and aggressiveness
    enemy_dx = nearest_enemy_x - ally_x
    enemy_dy = nearest_enemy_y - ally_y
    enemy_dist = jnp.sqrt(enemy_dx**2 + enemy_dy**2)
    
    # Shape assertions for enemy forces
    assert enemy_dx.shape == (batch_size, num_agents), f"enemy_dx shape {enemy_dx.shape} != {(batch_size, num_agents)}"
    assert enemy_dy.shape == (batch_size, num_agents), f"enemy_dy shape {enemy_dy.shape} != {(batch_size, num_agents)}"
    assert enemy_dist.shape == (batch_size, num_agents), f"enemy_dist shape {enemy_dist.shape} != {(batch_size, num_agents)}"
    
    # Normalize enemy forces
    enemy_dx = enemy_dx / (enemy_dist + 1e-5)
    enemy_dy = enemy_dy / (enemy_dist + 1e-5)
    
    # Determine attack/retreat behavior based on health
    health_factor = (ally_health - RETREAT_THRESHOLD) / (ATTACK_THRESHOLD - RETREAT_THRESHOLD)
    health_factor = jnp.clip(health_factor, 0, 1)
    
    # Shape assertions for health factors
    assert health_factor.shape == (batch_size, num_agents), f"health_factor shape {health_factor.shape} != {(batch_size, num_agents)}"
    
    # Blend between formation and enemy forces based on health and aggressiveness
    enemy_weight = health_factor * AGGRESSIVENESS
    formation_weight = 1 - enemy_weight
    
    # Shape assertions for weights
    assert enemy_weight.shape == (batch_size, num_agents), f"enemy_weight shape {enemy_weight.shape} != {(batch_size, num_agents)}"
    assert formation_weight.shape == (batch_size, num_agents), f"formation_weight shape {formation_weight.shape} != {(batch_size, num_agents)}"
    
    # Combine forces - ensure all arrays have shape [batch, ally]
    dx = formation_weight * formation_dx + enemy_weight * enemy_dx
    dy = formation_weight * formation_dy + enemy_weight * enemy_dy
    
    # Shape assertions for combined forces
    assert dx.shape == (batch_size, num_agents), f"dx shape {dx.shape} != {(batch_size, num_agents)}"
    assert dy.shape == (batch_size, num_agents), f"dy shape {dy.shape} != {(batch_size, num_agents)}"
    
    # Normalize combined forces
    force_mag = jnp.sqrt(dx**2 + dy**2)
    dx = dx / (force_mag + 1e-5)
    dy = dy / (force_mag + 1e-5)
    
    # Scale by max speed
    x_action = dx * MAX_SPEED
    y_action = dy * MAX_SPEED
    
    # Apply smoothness to velocity changes
    x_action = SMOOTHNESS * x_action + (1 - SMOOTHNESS) * ally_vx
    y_action = SMOOTHNESS * y_action + (1 - SMOOTHNESS) * ally_vy
    
    # Apply damping
    x_action = x_action - ally_vx * DAMPING
    y_action = y_action - ally_vy * DAMPING
    
    # Final shape assertions
    assert x_action.shape == (batch_size, num_agents), f"x_action shape {x_action.shape} != {(batch_size, num_agents)}"
    assert y_action.shape == (batch_size, num_agents), f"y_action shape {y_action.shape} != {(batch_size, num_agents)}"
    
    return x_action, y_action
