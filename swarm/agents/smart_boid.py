import jax
import jax.numpy as jnp

SEPARATION_RADIUS = 0.08
PERCEPTION_RADIUS = 0.2
SEPARATION_WEIGHT = 0.05
ALIGNMENT_WEIGHT = 0.05
COHESION_WEIGHT = 0.06
DAMPING = 0.1

CHASE_RADIUS = 0.25
CHASE_WEIGHT = 0.08
FLEE_RADIUS = 0.15
FLEE_WEIGHT = 0.06
MIN_GROUP_SIZE = 2
HEALTH_THRESHOLD = 0.3

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
    """Smart boid agent that adapts its behavior based on health and group size."""
    x_action = jnp.zeros_like(ally_x)
    y_action = jnp.zeros_like(ally_y)
    
    x_action -= ally_vx * DAMPING
    y_action -= ally_vy * DAMPING
    
    ally_dx = ally_x[:, None, :] - ally_x[:, :, None]
    ally_dy = ally_y[:, None, :] - ally_y[:, :, None]
    
    ally_dx = jnp.where(ally_dx > 0.5, ally_dx - 1.0, ally_dx)
    ally_dx = jnp.where(ally_dx < -0.5, ally_dx + 1.0, ally_dx)
    ally_dy = jnp.where(ally_dy > 0.5, ally_dy - 1.0, ally_dy)
    ally_dy = jnp.where(ally_dy < -0.5, ally_dy + 1.0, ally_dy)
    
    ally_dist = jnp.sqrt(ally_dx**2 + ally_dy**2)
    
    separation_mask = jnp.tril(ally_dist < SEPARATION_RADIUS, -1)
    x_action += (ally_dx * separation_mask).sum(axis=1) * SEPARATION_WEIGHT
    y_action += (ally_dy * separation_mask).sum(axis=1) * SEPARATION_WEIGHT
    
    alignment_mask = jnp.tril((ally_dist < PERCEPTION_RADIUS) & (ally_dist > SEPARATION_RADIUS), -1)
    vx_total = jnp.sum(ally_vx[:, None, :] * alignment_mask, axis=1)
    vy_total = jnp.sum(ally_vy[:, None, :] * alignment_mask, axis=1)
    alignment_count = jnp.sum(alignment_mask, axis=1)
    vx_avg = jnp.where(alignment_count > 0, vx_total / alignment_count, ally_vx)
    vy_avg = jnp.where(alignment_count > 0, vy_total / alignment_count, ally_vy)
    x_action += (vx_avg - ally_vx) * ALIGNMENT_WEIGHT
    y_action += (vy_avg - ally_vy) * ALIGNMENT_WEIGHT
    
    cohesion_mask = jnp.tril(ally_dist < PERCEPTION_RADIUS, -1)
    x_total = jnp.sum(ally_x[:, None, :] * cohesion_mask, axis=1)
    y_total = jnp.sum(ally_y[:, None, :] * cohesion_mask, axis=1)
    cohesion_count = jnp.sum(cohesion_mask, axis=1)
    x_avg = jnp.where(cohesion_count > 0, x_total / cohesion_count, ally_x)
    y_avg = jnp.where(cohesion_count > 0, y_total / cohesion_count, ally_y)
    x_action += (x_avg - ally_x) * COHESION_WEIGHT
    y_action += (y_avg - ally_y) * COHESION_WEIGHT
    
    enemy_dx = ally_x[:, None, :] - enemy_x[:, :, None]
    enemy_dy = ally_y[:, None, :] - enemy_y[:, :, None]
    
    enemy_dx = jnp.where(enemy_dx > 0.5, enemy_dx - 1.0, enemy_dx)
    enemy_dx = jnp.where(enemy_dx < -0.5, enemy_dx + 1.0, enemy_dx)
    enemy_dy = jnp.where(enemy_dy > 0.5, enemy_dy - 1.0, enemy_dy)
    enemy_dy = jnp.where(enemy_dy < -0.5, enemy_dy + 1.0, enemy_dy)
    
    enemy_dist = jnp.sqrt(enemy_dx**2 + enemy_dy**2)
    
    min_enemy_dist = jnp.min(enemy_dist, axis=1)
    closest_enemy_idx = jnp.argmin(enemy_dist, axis=1)
    
    batch_idx = jnp.arange(ally_x.shape[0])[:, None]
    enemy_idx = closest_enemy_idx
    agent_idx = jnp.arange(ally_x.shape[1])[None, :]
    
    closest_enemy_dx = enemy_dx[batch_idx, enemy_idx, agent_idx]
    closest_enemy_dy = enemy_dy[batch_idx, enemy_idx, agent_idx]
    
    group_size = jnp.sum(ally_dist < PERCEPTION_RADIUS, axis=1)
    group_advantage = group_size > MIN_GROUP_SIZE
    health_aggression = ally_health > HEALTH_THRESHOLD
    
    chase_mask = (min_enemy_dist < CHASE_RADIUS) & group_advantage & health_aggression
    x_action += -closest_enemy_dx * chase_mask * CHASE_WEIGHT
    y_action += -closest_enemy_dy * chase_mask * CHASE_WEIGHT
    
    flee_mask = (min_enemy_dist < FLEE_RADIUS) & (~group_advantage | ~health_aggression)
    x_action += closest_enemy_dx * flee_mask * FLEE_WEIGHT
    y_action += closest_enemy_dy * flee_mask * FLEE_WEIGHT
    
    return x_action, y_action
