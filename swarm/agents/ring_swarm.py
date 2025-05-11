import jax
import jax.numpy as jnp

FORMATION_CENTER_X = 0.5
FORMATION_CENTER_Y = 0.5
FORMATION_RADIUS = 0.2
FORMATION_WEIGHT = 0.06
VELOCITY_WEIGHT = 0.08
DAMPING = 0.1
RETREAT_HEALTH_THRESHOLD = 0.4
RETREAT_WEIGHT = 0.1

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
    """Ring swarm agent that forms a static ring formation with health-based retreat."""
    x_action = jnp.zeros_like(ally_x)
    y_action = jnp.zeros_like(ally_y)
    
    dx = ally_x - FORMATION_CENTER_X
    dy = ally_y - FORMATION_CENTER_Y
    
    dx = jnp.where(dx > 0.5, dx - 1.0, dx)
    dx = jnp.where(dx < -0.5, dx + 1.0, dx)
    dy = jnp.where(dy > 0.5, dy - 1.0, dy)
    dy = jnp.where(dy < -0.5, dy + 1.0, dy)
    
    num_agents = ally_x.shape[1]
    agent_indices = jnp.arange(num_agents)
    target_angles = 2 * jnp.pi * agent_indices / num_agents
    
    target_dx = FORMATION_RADIUS * jnp.cos(target_angles)
    target_dy = FORMATION_RADIUS * jnp.sin(target_angles)
    
    target_vx = target_dx - dx
    target_vy = target_dy - dy
    
    target_speed = jnp.sqrt(target_vx**2 + target_vy**2)
    target_vx = jnp.where(target_speed > 0, target_vx / target_speed, 0)
    target_vy = jnp.where(target_speed > 0, target_vy / target_speed, 0)
    
    formation_dx = target_dx - dx
    formation_dy = target_dy - dy
    
    velocity_match_x = target_vx - ally_vx
    velocity_match_y = target_vy - ally_vy
    
    retreat_dx = -dx
    retreat_dy = -dy
    
    low_health_mask = ally_health < RETREAT_HEALTH_THRESHOLD
    retreat_scale = (RETREAT_HEALTH_THRESHOLD - ally_health) / RETREAT_HEALTH_THRESHOLD
    
    x_action += formation_dx * FORMATION_WEIGHT * (1 - low_health_mask)
    y_action += formation_dy * FORMATION_WEIGHT * (1 - low_health_mask)
    x_action += velocity_match_x * VELOCITY_WEIGHT * (1 - low_health_mask)
    y_action += velocity_match_y * VELOCITY_WEIGHT * (1 - low_health_mask)
    x_action += retreat_dx * RETREAT_WEIGHT * low_health_mask * retreat_scale
    y_action += retreat_dy * RETREAT_WEIGHT * low_health_mask * retreat_scale
    
    x_action -= ally_vx * DAMPING
    y_action -= ally_vy * DAMPING
    
    return x_action, y_action
