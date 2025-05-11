import jax
import jax.numpy as jnp

RING_CENTER_X = 0.5
RING_CENTER_Y = 0.5
RING_RADIUS = 0.3
MIN_SPACING = 0.1
RETREAT_HEALTH_THRESHOLD = 0.5
RETREAT_SPEED = 0.01
FORMATION_WEIGHT = 0.02
RETREAT_WEIGHT = 0.03
RANDOM_WEIGHT = 0.001

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
    """Clusters agent that forms multiple independent groups for coordinated movement."""
    x_action = jnp.zeros_like(ally_vx)
    y_action = jnp.zeros_like(ally_vy)
    
    dx = ally_x - RING_CENTER_X
    dy = ally_y - RING_CENTER_Y
    
    dx = jnp.where(dx > 0.5, dx - 1.0, dx)
    dx = jnp.where(dx < -0.5, dx + 1.0, dx)
    dy = jnp.where(dy > 0.5, dy - 1.0, dy)
    dy = jnp.where(dy < -0.5, dy + 1.0, dy)
    
    dist = jnp.sqrt(dx**2 + dy**2)
    angle = jnp.arctan2(dy, dx)
    
    num_agents = ally_x.shape[1]
    target_angles = jnp.linspace(0, 2 * jnp.pi, num_agents, endpoint=False)
    target_dx = RING_RADIUS * jnp.cos(target_angles)
    target_dy = RING_RADIUS * jnp.sin(target_angles)
    
    formation_dx = target_dx - dx
    formation_dy = target_dy - dy
    x_action += formation_dx * FORMATION_WEIGHT
    y_action += formation_dy * FORMATION_WEIGHT
    
    low_health_mask = ally_health < RETREAT_HEALTH_THRESHOLD
    retreat_scale = (RETREAT_HEALTH_THRESHOLD - ally_health) / RETREAT_HEALTH_THRESHOLD
    
    retreat_dx = -dx / (dist + 1e-6) * RETREAT_SPEED
    retreat_dy = -dy / (dist + 1e-6) * RETREAT_SPEED
    
    x_action += retreat_dx * low_health_mask * retreat_scale * RETREAT_WEIGHT
    y_action += retreat_dy * low_health_mask * retreat_scale * RETREAT_WEIGHT
    
    agent_dx = ally_x[:, None, :] - ally_x[:, :, None]
    agent_dy = ally_y[:, None, :] - ally_y[:, :, None]
    
    agent_dx = jnp.where(agent_dx > 0.5, agent_dx - 1.0, agent_dx)
    agent_dx = jnp.where(agent_dx < -0.5, agent_dx + 1.0, agent_dx)
    agent_dy = jnp.where(agent_dy > 0.5, agent_dy - 1.0, agent_dy)
    agent_dy = jnp.where(agent_dy < -0.5, agent_dy + 1.0, agent_dy)
    
    agent_dist = jnp.sqrt(agent_dx**2 + agent_dy**2)
    
    too_close = (agent_dist < MIN_SPACING) & (agent_dist > 0)
    repulsion_dx = agent_dx * too_close
    repulsion_dy = agent_dy * too_close
    
    x_action += jnp.sum(repulsion_dx, axis=2) * FORMATION_WEIGHT
    y_action += jnp.sum(repulsion_dy, axis=2) * FORMATION_WEIGHT
    
    xkey, ykey, _ = jax.random.split(key, 3)
    x_action += jax.random.uniform(xkey, ally_x.shape, minval=-RANDOM_WEIGHT, maxval=RANDOM_WEIGHT)
    y_action += jax.random.uniform(ykey, ally_y.shape, minval=-RANDOM_WEIGHT, maxval=RANDOM_WEIGHT)
    
    return x_action, y_action 