import jax
import jax.numpy as jnp

LINE_LENGTH: float = 0.32
CURVE_STRENGTH: float = 0.18
FORMATION_WEIGHT: float = 1.1

ADVANCE_SPEED: float = 0.16
RETREAT_SPEED: float = 0.19
DAMPING: float = 0.07

ENGAGEMENT_DISTANCE: float = 0.28
MIN_GROUP_SIZE: int = 2
HEALTH_AGGRESSION: float = 1.35

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
    state_t = t
    batch_size = ally_x.shape[0]
    num_agents = ally_x.shape[1]

    ally_com_x = jnp.mean(ally_x, axis=1, keepdims=True)
    ally_com_y = jnp.mean(ally_y, axis=1, keepdims=True)
    enemy_com_x = jnp.mean(enemy_x, axis=1, keepdims=True)
    enemy_com_y = jnp.mean(enemy_y, axis=1, keepdims=True)

    dx = enemy_com_x - ally_com_x
    dy = enemy_com_y - ally_com_y
        
    dx = jnp.where(dx > 0.5, dx - 1.0, dx)
    dx = jnp.where(dx < -0.5, dx + 1.0, dx)
    dy = jnp.where(dy > 0.5, dy - 1.0, dy)
    dy = jnp.where(dy < -0.5, dy + 1.0, dy)
    
    dist = jnp.sqrt(dx**2 + dy**2) + 1e-5
    dx = dx / dist
    dy = dy / dist
    
    t = jnp.linspace(-1.0, 1.0, num_agents)
    t = jnp.tile(t[None, :], (batch_size, 1))

    perp_x = -dy
    perp_y = dx
    
    formation_scale = jnp.minimum(1.0, 1.0 + state_t * 0.1)
    formation_scale = jnp.reshape(formation_scale, (batch_size, 1))
    
    line_x = jnp.multiply(t, perp_x)
    line_x = jnp.multiply(line_x, LINE_LENGTH)
    line_x = jnp.multiply(line_x, formation_scale)
    
    line_y = jnp.multiply(t, perp_y)
    line_y = jnp.multiply(line_y, LINE_LENGTH)
    line_y = jnp.multiply(line_y, formation_scale)
        
    curve = CURVE_STRENGTH * (1.0 - t**2)
    dx_broad = jnp.broadcast_to(dx, (batch_size, num_agents))
    dy_broad = jnp.broadcast_to(dy, (batch_size, num_agents))
    
    curve_x = jnp.multiply(curve, dx_broad)
    curve_x = jnp.multiply(curve_x, LINE_LENGTH)
    curve_x = jnp.multiply(curve_x, formation_scale)
    
    curve_y = jnp.multiply(curve, dy_broad)
    curve_y = jnp.multiply(curve_y, LINE_LENGTH)
    curve_y = jnp.multiply(curve_y, formation_scale)
        
    target_x = ally_com_x + line_x + curve_x
    target_y = ally_com_y + line_y + curve_y

    formation_dx = (target_x - ally_x) * FORMATION_WEIGHT
    formation_dy = (target_y - ally_y) * FORMATION_WEIGHT
    
    enemy_dx = ally_x[:, None, :] - enemy_x[:, :, None]
    enemy_dy = ally_y[:, None, :] - enemy_y[:, :, None]
    
    enemy_dx = jnp.where(enemy_dx > 0.5, enemy_dx - 1.0, enemy_dx)
    enemy_dx = jnp.where(enemy_dx < -0.5, enemy_dx + 1.0, enemy_dx)
    enemy_dy = jnp.where(enemy_dy > 0.5, enemy_dy - 1.0, enemy_dy)
    enemy_dy = jnp.where(enemy_dy < -0.5, enemy_dy + 1.0, enemy_dy)
    
    enemy_dist = jnp.sqrt(enemy_dx**2 + enemy_dy**2)
    
    nearby_enemies = jnp.sum(enemy_dist < ENGAGEMENT_DISTANCE, axis=1)
    
    should_advance = nearby_enemies >= MIN_GROUP_SIZE
    move_speed = jnp.where(should_advance, 
                            ADVANCE_SPEED * ally_health * HEALTH_AGGRESSION,
                            -RETREAT_SPEED)
    
    combat_dx = dx.reshape(batch_size, 1) * move_speed
    combat_dy = dy.reshape(batch_size, 1) * move_speed

    dx = formation_dx + 0.3 * combat_dx
    dy = formation_dy + 0.3 * combat_dy
    
    dx = dx - ally_vx * DAMPING
    dy = dy - ally_vy * DAMPING
        
    return dx, dy
