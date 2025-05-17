import jax
import jax.numpy as jnp
from jax import jit, vmap, lax

@jit
def act(
    t,
    key,
    ally_x,
    ally_y,
    ally_vx,
    ally_vy,
    ally_health,
    enemy_x,
    enemy_y,
    enemy_vx,
    enemy_vy,
    enemy_health,
):
    batch_size, num_agents = ally_x.shape
    num_enemies = enemy_x.shape[1]

    # Constants
    MOVE_SPEED = 0.7
    AVOID_RADIUS = 1.5
    ENGAGE_RADIUS = 5.0
    HEALTH_THRESHOLD = 0.5
    REGENERATION_RATE = 0.02
    MAX_HEALTH = 1.0
    SAFE_ZONE_RADIUS = 15.0
    SAFE_ZONE_CENTER = jnp.array([0.0, 0.0])
    SLOW_DOWN_DISTANCE = 8.0
    ENGAGE_FORCE = 0.8
    ESCAPE_FORCE = 0.2
    AVOID_FORCE = 1.2
    SLOW_DOWN_FORCE = 0.5
    FORMATION_RADIUS = 8.0
    FORMATION_FORCE = 0.5
    MAX_ACCELERATION = 1.0

    # Compute relative positions to enemies
    ally_to_enemy_dx = enemy_x[:, None, :] - ally_x[:, :, None]  # (B, A, E)
    ally_to_enemy_dy = enemy_y[:, None, :] - ally_y[:, :, None]  # (B, A, E)
    dist_sq = ally_to_enemy_dx**2 + ally_to_enemy_dy**2 + 1e-8  # (B, A, E)

    # Find the closest enemy for each agent
    min_dist_idx = jnp.argmin(dist_sq, axis=2)  # (B, A)
    min_dist_dx = jnp.take_along_axis(ally_to_enemy_dx, min_dist_idx[:, :, None], axis=2).squeeze(2)  # (B, A)
    min_dist_dy = jnp.take_along_axis(ally_to_enemy_dy, min_dist_idx[:, :, None], axis=2).squeeze(2)  # (B, A)

    # Normalize direction to closest enemy
    dist = jnp.sqrt(min_dist_dx**2 + min_dist_dy**2)
    direction_x = min_dist_dx / (dist + 1e-8)
    direction_y = min_dist_dy / (dist + 1e-8)

    # Engagement and avoidance masks
    engage_mask = dist < ENGAGE_RADIUS
    avoid_mask = dist < AVOID_RADIUS
    slow_mask = (dist < SLOW_DOWN_DISTANCE) & (dist > AVOID_RADIUS)

    # Avoid close allies
    ally_to_ally_dx = ally_x[:, :, None] - ally_x[:, None, :]  # (B, A, A)
    ally_to_ally_dy = ally_y[:, :, None] - ally_y[:, None, :]  # (B, A, A)
    ally_dist_sq = ally_to_ally_dx**2 + ally_to_ally_dy**2 + 1e-8  # (B, A, A)
    ally_mask = (ally_dist_sq < (AVOID_RADIUS ** 2)).any(axis=2)  # (B, A)

    # Avoid close enemies (for all agents)
    enemy_to_ally_dx = ally_x[:, :, None] - enemy_x[:, None, :]  # (B, A, E)
    enemy_to_ally_dy = ally_y[:, :, None] - enemy_y[:, None, :]  # (B, A, E)
    enemy_dist_sq = enemy_to_ally_dx**2 + enemy_to_ally_dy**2 + 1e-8  # (B, A, E)
    enemy_mask = (enemy_dist_sq < (AVOID_RADIUS ** 2)).any(axis=2)  # (B, A)

    # Combine avoidance masks
    avoid_mask = ally_mask | enemy_mask | avoid_mask  # (B, A)

    # Compute formation center
    formation_center_x = jnp.mean(ally_x, axis=1, keepdims=True)  # (B, 1)
    formation_center_y = jnp.mean(ally_y, axis=1, keepdims=True)  # (B, 1)
    formation_to_agent_dx = formation_center_x - ally_x  # (B, A)
    formation_to_agent_dy = formation_center_y - ally_y  # (B, A)
    formation_dist_sq = formation_to_agent_dx**2 + formation_to_agent_dy**2 + 1e-8
    formation_mask = formation_dist_sq > (FORMATION_RADIUS ** 2)  # (B, A)
    formation_dir_x = formation_to_agent_dx / (jnp.sqrt(formation_dist_sq) + 1e-8)
    formation_dir_y = formation_to_agent_dy / (jnp.sqrt(formation_dist_sq) + 1e-8)

    # Health-based behavior
    low_health = ally_health < HEALTH_THRESHOLD
    health_factor = jnp.where(low_health, 0.0, 1.0)
    regen_factor = jnp.where(low_health, 1.0, 0.0)

    # Initialize dvx and dvy
    dvx = jnp.zeros((batch_size, num_agents))
    dvy = jnp.zeros((batch_size, num_agents))

    # Move toward enemy with weighted behavior
    dvx += direction_x * MOVE_SPEED * (1 - avoid_mask) * health_factor
    dvy += direction_y * MOVE_SPEED * (1 - avoid_mask) * health_factor

    # Encourage movement toward enemy when in range
    dvx += direction_x * ENGAGE_FORCE * engage_mask * (1 - regen_factor)
    dvy += direction_y * ENGAGE_FORCE * engage_mask * (1 - regen_factor)

    # Slow down when approaching
    dvx *= (1 - slow_mask * SLOW_DOWN_FORCE * (1 - regen_factor))
    dvy *= (1 - slow_mask * SLOW_DOWN_FORCE * (1 - regen_factor))

    # If low on health, move toward safe zone
    safe_to_ally_dx = SAFE_ZONE_CENTER[0] - ally_x
    safe_to_ally_dy = SAFE_ZONE_CENTER[1] - ally_y
    safe_dist = jnp.sqrt(safe_to_ally_dx**2 + safe_to_ally_dy**2)
    safe_dir_x = safe_to_ally_dx / (safe_dist + 1e-8)
    safe_dir_y = safe_to_ally_dy / (safe_dist + 1e-8)
    dvx += safe_dir_x * ESCAPE_FORCE * low_health
    dvy += safe_dir_y * ESCAPE_FORCE * low_health

    # Pull agents toward formation center
    dvx += formation_dir_x * FORMATION_FORCE * formation_mask
    dvy += formation_dir_y * FORMATION_FORCE * formation_mask

    # Limit acceleration
    acceleration = jnp.sqrt(dvx**2 + dvy**2)
    dvx = dvx * (MAX_ACCELERATION / (acceleration + 1e-8))
    dvy = dvy * (MAX_ACCELERATION / (acceleration + 1e-8))

    # Regenerate health over time
    ally_health = ally_health + REGENERATION_RATE * (1 - ally_health) * (1 - regen_factor)
    ally_health = jnp.clip(ally_health, 0.0, MAX_HEALTH)

    return dvx, dvy
