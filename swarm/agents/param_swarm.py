import jax
import jax.numpy as jnp

FORMATION_SCALE = 0.7
FORMATION_SHAPE = 0.3
AGGRESSIVENESS = 0.6
SMOOTHNESS = 0.8
ATTACK_THRESHOLD = 0.3
RETREAT_THRESHOLD = 0.2

MAX_SPEED = 0.01
DAMPING = 0.1

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
    """Parametric swarm agent with tunable behavior."""
    batch_size, num_agents = ally_x.shape

    ally_com_x = jnp.mean(ally_x, axis=1, keepdims=True)
    ally_com_y = jnp.mean(ally_y, axis=1, keepdims=True)

    rel_x = ally_x - ally_com_x
    rel_y = ally_y - ally_com_y
    
    angles = jnp.linspace(0, 2 * jnp.pi, num_agents, endpoint=False)
    circle_x = jnp.cos(angles) * FORMATION_SCALE
    circle_y = jnp.sin(angles) * FORMATION_SCALE
    
    line_x = jnp.linspace(-FORMATION_SCALE, FORMATION_SCALE, num_agents)
    line_y = jnp.zeros_like(line_x)
    
    target_x = (1 - FORMATION_SHAPE) * circle_x + FORMATION_SHAPE * line_x
    target_y = (1 - FORMATION_SHAPE) * circle_y + FORMATION_SHAPE * line_y
    
    formation_dx = target_x - rel_x
    formation_dy = target_y - rel_y
    
    dx = enemy_x[:, None, :] - ally_x[:, :, None]
    dy = enemy_y[:, None, :] - ally_y[:, :, None]
    distances = jnp.sqrt(dx**2 + dy**2)
    nearest_enemy_idx = jnp.argmin(distances, axis=-1)

    batch_indices = jnp.arange(batch_size)[:, None]
    nearest_enemy_x = enemy_x[batch_indices, nearest_enemy_idx]
    nearest_enemy_y = enemy_y[batch_indices, nearest_enemy_idx]
    
    enemy_dx = nearest_enemy_x - ally_x
    enemy_dy = nearest_enemy_y - ally_y
    enemy_dist = jnp.sqrt(enemy_dx**2 + enemy_dy**2)
    
    enemy_dx = enemy_dx / (enemy_dist + 1e-5)
    enemy_dy = enemy_dy / (enemy_dist + 1e-5)
    
    health_factor = (ally_health - RETREAT_THRESHOLD) / (ATTACK_THRESHOLD - RETREAT_THRESHOLD)
    health_factor = jnp.clip(health_factor, 0, 1)
    
    enemy_weight = health_factor * AGGRESSIVENESS
    formation_weight = 1 - enemy_weight
    
    dx = formation_weight * formation_dx + enemy_weight * enemy_dx
    dy = formation_weight * formation_dy + enemy_weight * enemy_dy
    
    force_mag = jnp.sqrt(dx**2 + dy**2)
    dx = dx / (force_mag + 1e-5)
    dy = dy / (force_mag + 1e-5)
    
    x_action = dx * MAX_SPEED
    y_action = dy * MAX_SPEED
    
    x_action = SMOOTHNESS * x_action + (1 - SMOOTHNESS) * ally_vx
    y_action = SMOOTHNESS * y_action + (1 - SMOOTHNESS) * ally_vy
    
    x_action = x_action - ally_vx * DAMPING
    y_action = y_action - ally_vy * DAMPING
    
    return x_action, y_action
