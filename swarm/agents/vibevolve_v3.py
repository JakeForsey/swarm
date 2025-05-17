import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial

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

    # Constants
    MOVE_SPEED = 0.5
    AVOID_RADIUS = 1.0
    AVOID_RADIUS_SLOW = 2.0
    HEALTH_THRESHOLD = 0.5
    REGENERATION_RATE = 0.02
    MIN_ATTACK_DISTANCE = 0.5
    MAX_ATTACK_DISTANCE = 3.0
    GROUP_RADIUS = 3.0
    MAX_STEER_FORCE = 0.2
    SEPARATION_WEIGHT = 0.5
    COHESION_WEIGHT = 0.4
    ALIGNMENT_WEIGHT = 0.3
    ATTACK_WEIGHT = 0.6
    CHASE_WEIGHT = 0.5
    REGEN_FOCUSED_HEALTH_THRESHOLD = 0.3
    REGEN_FOCUSED_DISTANCE = 3.0
    ATTACK_FOCUSED_DISTANCE = 1.5

    # Compute relative positions to enemies
    ally_to_enemy_dx = enemy_x[:, None, :] - ally_x[:, :, None]  # (B, A, E)
    ally_to_enemy_dy = enemy_y[:, None, :] - ally_y[:, :, None]  # (B, A, E)
    dist_sq = ally_to_enemy_dx**2 + ally_to_enemy_dy**2 + 1e-8
    dist = jnp.sqrt(dist_sq)

    # Find the index of the closest enemy for each agent
    min_dist_idx = jnp.argmin(dist_sq, axis=2)
    min_dist_dx = jnp.take_along_axis(ally_to_enemy_dx, min_dist_idx[:, :, None], axis=2).squeeze(2)
    min_dist_dy = jnp.take_along_axis(ally_to_enemy_dy, min_dist_idx[:, :, None], axis=2).squeeze(2)
    min_dist = jnp.sqrt(min_dist_dx**2 + min_dist_dy**2 + 1e-8)

    # Compute direction toward the closest enemy
    direction_x = min_dist_dx / (min_dist + 1e-8)
    direction_y = min_dist_dy / (min_dist + 1e-8)

    # Determine behavior based on distance to enemy
    is_attacking = (min_dist < MAX_ATTACK_DISTANCE) & (min_dist > MIN_ATTACK_DISTANCE)
    is_chasing = (min_dist > MAX_ATTACK_DISTANCE) & (min_dist < MAX_ATTACK_DISTANCE * 1.5)
    is_avoiding = min_dist < AVOID_RADIUS
    is_avoiding_slow = (min_dist < AVOID_RADIUS_SLOW) & (~is_avoiding)

    # Compute avoidance factors
    avoid_factor = jnp.where(is_avoiding, 0.1, 1.0)
    avoid_factor = jnp.where(is_avoiding_slow, 0.5, avoid_factor)

    # Compute health-based behavior
    low_health = ally_health < HEALTH_THRESHOLD
    health_factor = jnp.where(low_health, 0.0, 1.0)
    regen_focused = ally_health < REGEN_FOCUSED_HEALTH_THRESHOLD
    attack_focused = (ally_health > 0.7) & (min_dist < ATTACK_FOCUSED_DISTANCE)

    # Avoid close allies
    ally_to_ally_dx = ally_x[:, :, None] - ally_x[:, None, :]
    ally_to_ally_dy = ally_y[:, :, None] - ally_y[:, None, :]
    ally_dist_sq = ally_to_ally_dx**2 + ally_to_ally_dy**2
    ally_mask = ally_dist_sq < (AVOID_RADIUS ** 2)
    ally_mask = jnp.any(ally_mask, axis=2)
    avoid_factor = jnp.where(ally_mask, 0.1, avoid_factor)

    # Avoid close enemies
    enemy_to_ally_dx = ally_x[:, :, None] - enemy_x[:, None, :]
    enemy_to_ally_dy = ally_y[:, :, None] - enemy_y[:, None, :]
    enemy_dist_sq = enemy_to_ally_dx**2 + enemy_to_ally_dy**2
    enemy_mask = enemy_dist_sq < (AVOID_RADIUS ** 2)
    enemy_mask = jnp.any(enemy_mask, axis=2)
    avoid_factor = jnp.where(enemy_mask, 0.1, avoid_factor)

    # Compute group center
    group_center_x = jnp.mean(ally_x, axis=1, keepdims=True)
    group_center_y = jnp.mean(ally_y, axis=1, keepdims=True)

    # Compute cohesion toward group center
    cohesion_dx = group_center_x - ally_x
    cohesion_dy = group_center_y - ally_y
    cohesion_dist = jnp.sqrt(cohesion_dx**2 + cohesion_dy**2 + 1e-8)
    cohesion_dir_x = cohesion_dx / (cohesion_dist + 1e-8)
    cohesion_dir_y = cohesion_dy / (cohesion_dist + 1e-8)
    cohesion_weight = jnp.where(ally_health > HEALTH_THRESHOLD, 1.0, 0.5)
    cohesion_x = cohesion_dir_x * COHESION_WEIGHT * cohesion_weight
    cohesion_y = cohesion_dir_y * COHESION_WEIGHT * cohesion_weight

    # Compute separation from nearby allies
    ally_to_ally_dx = ally_x[:, :, None] - ally_x[:, None, :]
    ally_to_ally_dy = ally_y[:, :, None] - ally_y[:, None, :]
    ally_dist = jnp.sqrt(ally_to_ally_dx**2 + ally_to_ally_dy**2 + 1e-8)
    separation_factor = jnp.where(ally_dist < AVOID_RADIUS, 1.0 / (ally_dist + 1e-8), 0.0)
    separation_x = jnp.sum(ally_to_ally_dx * separation_factor, axis=1)
    separation_y = jnp.sum(ally_to_ally_dy * separation_factor, axis=1)
    separation_dist = jnp.sqrt(separation_x**2 + separation_y**2 + 1e-8)
    separation_dir_x = separation_x / (separation_dist + 1e-8)
    separation_dir_y = separation_y / (separation_dist + 1e-8)
    separation_x = separation_dir_x * SEPARATION_WEIGHT
    separation_y = separation_dir_y * SEPARATION_WEIGHT

    # Compute alignment with nearby allies
    ally_vx_avg = jnp.mean(ally_vx, axis=1, keepdims=True)
    ally_vy_avg = jnp.mean(ally_vy, axis=1, keepdims=True)
    alignment_diff_x = ally_vx_avg - ally_vx
    alignment_diff_y = ally_vy_avg - ally_vy
    alignment_weight = jnp.where(ally_health > HEALTH_THRESHOLD, 1.0, 0.5)
    alignment_x = alignment_diff_x * ALIGNMENT_WEIGHT * alignment_weight
    alignment_y = alignment_diff_y * ALIGNMENT_WEIGHT * alignment_weight

    # Combine steering behaviors
    steering_x = separation_x + cohesion_x + alignment_x
    steering_y = separation_y + cohesion_y + alignment_y

    # Add attack behavior
    attack_x = direction_x * is_attacking * ATTACK_WEIGHT
    attack_y = direction_y * is_attacking * ATTACK_WEIGHT
    steering_x += attack_x
    steering_y += attack_y

    # Add chase behavior
    chase_x = direction_x * is_chasing * CHASE_WEIGHT
    chase_y = direction_y * is_chasing * CHASE_WEIGHT
    steering_x += chase_x
    steering_y += chase_y

    # Add regen-focused behavior
    regen_dir_x = jnp.zeros_like(steering_x)
    regen_dir_y = jnp.zeros_like(steering_y)
    regen_dir_x = jnp.where(regen_focused, group_center_x - ally_x, regen_dir_x)
    regen_dir_y = jnp.where(regen_focused, group_center_y - ally_y, regen_dir_y)
    regen_dir_mag = jnp.sqrt(regen_dir_x**2 + regen_dir_y**2 + 1e-8)
    regen_dir_x /= (regen_dir_mag + 1e-8)
    regen_dir_y /= (regen_dir_mag + 1e-8)
    regen_dir_x *= 0.3
    regen_dir_y *= 0.3
    steering_x += regen_dir_x
    steering_y += regen_dir_y

    # Add attack-focused behavior
    attack_dir_x = direction_x * attack_focused * 1.2
    attack_dir_y = direction_y * attack_focused * 1.2
    steering_x += attack_dir_x
    steering_y += attack_dir_y

    # Normalize steering vector
    steering_mag = jnp.sqrt(steering_x**2 + steering_y**2 + 1e-8)
    steering_x = steering_x / (steering_mag + 1e-8)
    steering_y = steering_y / (steering_mag + 1e-8)

    # Scale the steering by speed and avoidance factor
    movement_dir_x = steering_x * MOVE_SPEED * avoid_factor * health_factor
    movement_dir_y = steering_y * MOVE_SPEED * avoid_factor * health_factor

    # Apply max steering force
    movement_dir_mag = jnp.sqrt(movement_dir_x**2 + movement_dir_y**2 + 1e-8)
    movement_dir_x = jnp.clip(movement_dir_x, -MAX_STEER_FORCE, MAX_STEER_FORCE)
    movement_dir_y = jnp.clip(movement_dir_y, -MAX_STEER_FORCE, MAX_STEER_FORCE)

    # Return the movement deltas
    dvx = movement_dir_x
    dvy = movement_dir_y

    return dvx, dvy
