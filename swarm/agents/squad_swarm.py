import jax
import jax.numpy as jnp

SQUAD_RADIUS = 0.15
FORMATION_WEIGHT = 0.08
VELOCITY_WEIGHT = 0.06
DAMPING = 0.15

CHASE_RADIUS = 0.2
CHASE_WEIGHT = 0.05
MIN_SQUAD_SIZE = 2
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
    """Squad swarm that forms multiple tight clusters to chase enemies while maintaining formation."""
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
    
    squad_mask = ally_dist < SQUAD_RADIUS
    
    squad_centers_x = jnp.zeros_like(ally_x)
    squad_centers_y = jnp.zeros_like(ally_y)
    
    for i in range(ally_x.shape[1]):
        squad_members = squad_mask[:, i, :]
        
        rel_x = ally_x - ally_x[:, i:i+1]
        rel_y = ally_y - ally_y[:, i:i+1]
        
        rel_x = jnp.where(rel_x > 0.5, rel_x - 1.0, rel_x)
        rel_x = jnp.where(rel_x < -0.5, rel_x + 1.0, rel_x)
        rel_y = jnp.where(rel_y > 0.5, rel_y - 1.0, rel_y)
        rel_y = jnp.where(rel_y < -0.5, rel_y + 1.0, rel_y)
        
        mean_rel_x = jnp.sum(rel_x * squad_members, axis=1, keepdims=True) / jnp.maximum(jnp.sum(squad_members, axis=1, keepdims=True), 1)
        mean_rel_y = jnp.sum(rel_y * squad_members, axis=1, keepdims=True) / jnp.maximum(jnp.sum(squad_members, axis=1, keepdims=True), 1)
        
        squad_centers_x = jnp.where(squad_members, ally_x[:, i:i+1] + mean_rel_x, squad_centers_x)
        squad_centers_y = jnp.where(squad_members, ally_y[:, i:i+1] + mean_rel_y, squad_centers_y)
    
    dx = squad_centers_x - ally_x
    dy = squad_centers_y - ally_y
    
    dx = jnp.where(dx > 0.5, dx - 1.0, dx)
    dx = jnp.where(dx < -0.5, dx + 1.0, dx)
    dy = jnp.where(dy > 0.5, dy - 1.0, dy)
    dy = jnp.where(dy < -0.5, dy + 1.0, dy)
    
    x_action += dx * FORMATION_WEIGHT
    y_action += dy * FORMATION_WEIGHT
    
    vx_expanded = ally_vx[:, None, :]
    vy_expanded = ally_vy[:, None, :]
    
    squad_vx = jnp.sum(vx_expanded * squad_mask, axis=2) / jnp.maximum(jnp.sum(squad_mask, axis=2), 1)
    squad_vy = jnp.sum(vy_expanded * squad_mask, axis=2) / jnp.maximum(jnp.sum(squad_mask, axis=2), 1)
    
    x_action += (squad_vx - ally_vx) * VELOCITY_WEIGHT
    y_action += (squad_vy - ally_vy) * VELOCITY_WEIGHT
    
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
    
    health_expanded = ally_health[:, None, :]
    
    squad_health = jnp.sum(health_expanded * squad_mask, axis=2) / jnp.maximum(jnp.sum(squad_mask, axis=2), 1)
    squad_size = jnp.sum(squad_mask, axis=2)
    
    chase_mask = (min_enemy_dist < CHASE_RADIUS) & (squad_health > HEALTH_THRESHOLD) & (squad_size >= MIN_SQUAD_SIZE)
    x_action += -closest_enemy_dx * chase_mask * CHASE_WEIGHT
    y_action += -closest_enemy_dy * chase_mask * CHASE_WEIGHT
    
    return x_action, y_action
