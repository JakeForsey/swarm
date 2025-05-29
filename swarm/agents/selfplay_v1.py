
import jax
import jax.numpy as jnp
from jax import jit, random

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

    # Normalize positions to [0, 1)
    ally_x_norm = (ally_x + 1.0) / 2.0
    ally_y_norm = (ally_y + 1.0) / 2.0
    enemy_x_norm = (enemy_x + 1.0) / 2.0
    enemy_y_norm = (enemy_y + 1.0) / 2.0

    # Compute pairwise distances between ally and enemy pieces
    dx = jnp.expand_dims(ally_x_norm, -1) - jnp.expand_dims(enemy_x_norm, -2)
    dy = jnp.expand_dims(ally_y_norm, -1) - jnp.expand_dims(enemy_y_norm, -2)
    dist = jnp.sqrt(dx**2 + dy**2 + 1e-8)

    # Find closest enemy for each ally piece
    min_dist_indices = jnp.argmin(dist, axis=-1)
    closest_enemy_x = jnp.take_along_axis(enemy_x_norm, min_dist_indices, axis=-1)
    closest_enemy_y = jnp.take_along_axis(enemy_y_norm, min_dist_indices, axis=-1)
    closest_enemy_vx = jnp.take_along_axis(enemy_vx, min_dist_indices, axis=-1)
    closest_enemy_vy = jnp.take_along_axis(enemy_vy, min_dist_indices, axis=-1)
    closest_enemy_health = jnp.take_along_axis(enemy_health, min_dist_indices, axis=-1)

    # Predict enemy position in the next step (assuming constant velocity)
    predicted_enemy_x = closest_enemy_x + closest_enemy_vx
    predicted_enemy_y = closest_enemy_y + closest_enemy_vy

    # Compute direction toward predicted enemy
    dx_to_enemy = predicted_enemy_x - ally_x_norm
    dy_to_enemy = predicted_enemy_y - ally_y_norm
    dist_to_enemy = jnp.sqrt(dx_to_enemy**2 + dy_to_enemy**2 + 1e-8)
    dir_x = dx_to_enemy / (dist_to_enemy + 1e-8)
    dir_y = dy_to_enemy / (dist_to_enemy + 1e-8)

    # Compute direction away from enemy
    dir_away_x = -dir_x
    dir_away_y = -dir_y

    # Compute direction toward center of mass
    mass = jnp.sum(ally_health, axis=-1, keepdims=True)
    center_x = jnp.sum(ally_x_norm * ally_health, axis=-1, keepdims=True) / mass
    center_y = jnp.sum(ally_y_norm * ally_health, axis=-1, keepdims=True) / mass
    dx_to_center = center_x - ally_x_norm
    dy_to_center = center_y - ally_y_norm
    dist_to_center = jnp.sqrt(dx_to_enemy**2 + dy_to_enemy**2 + 1e-8)
    dir_center_x = dx_to_center / (dist_to_center + 1e-8)
    dir_center_y = dy_to_center / (dist_to_center + 1e-8)

    # Health-based strategy switching
    health = ally_health
    attack_weight = jnp.clip((health - 0.7) / 0.3, 0.0, 1.0)  # Aggressive when health > 0.7
    defense_weight = jnp.clip((0.3 - health) / 0.3, 0.0, 1.0)  # Defensive when health < 0.3
    neutral_weight = 1.0 - attack_weight - defense_weight

    # Blend movement directions based on health
    dir_blend_x = (
        attack_weight * dir_x +
        defense_weight * dir_away_x +
        neutral_weight * (dir_x * 0.5 + dir_away_x * 0.5)
    )
    dir_blend_y = (
        attack_weight * dir_y +
        defense_weight * dir_away_y +
        neutral_weight * (dir_y * 0.5 + dir_away_y * 0.5)
    )

    # Add formation direction (center of mass)
    formation_weight = 0.2  # Weight of formation pull
    dir_blend_x += dir_center_x * formation_weight
    dir_blend_y += dir_center_y * formation_weight

    # Reduce acceleration toward close enemies (avoid collisions)
    close_enemies = (dist_to_enemy < 0.15)
    dir_blend_x = jnp.where(close_enemies, dir_blend_x * 0.5, dir_blend_x)
    dir_blend_y = jnp.where(close_enemies, dir_blend_y * 0.5, dir_blend_y)

    # Add repulsion force from nearby enemies
    repulsion_strength = jnp.clip(0.2 * (0.2 - dist_to_enemy), 0.0, 0.1)
    dir_blend_x -= dir_x * repulsion_strength
    dir_blend_y -= dir_y * repulsion_strength

    # Add randomness for exploration
    noise = random.uniform(key, (batch_size, num_agents), minval=-0.02, maxval=0.02)
    dir_blend_x += noise
    dir_blend_y += noise

    # Health-based velocity scaling
    velocity_scalar = 0.05 * (1.0 + jnp.clip(ally_health * 0.5, 0.0, 1.0))
    dvx = dir_blend_x * velocity_scalar
    dvy = dir_blend_y * velocity_scalar

    # Avoid overlapping with other ally pieces
    ally_positions = jnp.stack([ally_x_norm, ally_y_norm], axis=-1)
    dist_to_allies = jnp.sqrt(jnp.sum((jnp.expand_dims(ally_positions, -2) - jnp.expand_dims(ally_positions, -1))**2, axis=-1) + 1e-8)
    dist_to_allies = jnp.triu(dist_to_allies, 1)  # Ignore self
    too_close = (dist_to_allies < 0.15)
    too_close = jnp.any(too_close, axis=-1)
    dvx = jnp.where(too_close, dvx * 0.5, dvx)
    dvy = jnp.where(too_close, dvy * 0.5, dvy)

    # Regenerate health for surviving pieces
    ally_health = jnp.where(ally_health > 0.0, ally_health + 0.005, ally_health)
    ally_health = jnp.clip(ally_health, 0.0, 1.0)

    # If health is low, move randomly
    low_health_mask = (ally_health < 0.3)
    dvx = jnp.where(low_health_mask, random.uniform(key, (batch_size, num_agents), minval=-0.05, maxval=0.05), dvx)
    dvy = jnp.where(low_health_mask, random.uniform(key, (batch_size, num_agents), minval=-0.05, maxval=0.05), dvy)

    return dvx, dvy
