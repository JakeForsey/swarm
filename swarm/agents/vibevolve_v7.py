import jax
import jax.numpy as jnp
from jax import random as jrandom

@jax.jit
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

    # Stack positions and velocities for easier vectorization
    ally_positions = jnp.stack([ally_x, ally_y], axis=-1)  # (B, 32, 2)
    ally_velocity = jnp.stack([ally_vx, ally_vy], axis=-1)  # (B, 32, 2)
    enemy_positions = jnp.stack([enemy_x, enemy_y], axis=-1)  # (B, 32, 2)
    enemy_velocity = jnp.stack([enemy_vx, enemy_vy], axis=-1)  # (B, 32, 2)

    # Compute pairwise distances between ally and enemy pieces
    distances = jnp.linalg.norm(
        ally_positions[:, :, None, :] - enemy_positions[:, None, :, :], axis=-1
    )  # (B, 32, 32)

    # Weighted nearest enemy selection (prioritize low health enemies)
    enemy_health_expanded = enemy_health[:, None, :]  # (B, 1, 32)
    enemy_health_weight = 1.0 / (enemy_health_expanded + 1e-6)
    weighted_distances = distances * enemy_health_weight
    nearest_enemy_idx = jnp.argmin(weighted_distances, axis=-1)  # (B, 32)

    # Get nearest enemy info
    nearest_enemy_pos = jnp.take_along_axis(enemy_positions, nearest_enemy_idx[:, :, None], axis=1)  # (B, 32, 2)
    nearest_enemy_vel = jnp.take_along_axis(enemy_velocity, nearest_enemy_idx[:, :, None], axis=1)  # (B, 32, 2)
    nearest_enemy_health = jnp.take_along_axis(enemy_health, nearest_enemy_idx, axis=1)  # (B, 32)

    # Compute direction toward the nearest enemy
    direction = nearest_enemy_pos - ally_positions  # (B, 32, 2)
    norm = jnp.linalg.norm(direction, axis=-1, keepdims=True) + 1e-6
    direction = direction / norm

    # Compute distance and health factors
    distance_factor = 1.0 / (jnp.linalg.norm(direction, axis=-1) + 1e-6)
    health_factor = (1.0 - nearest_enemy_health) * 0.8
    movement_intensity = jnp.tanh(0.5 * (distance_factor * health_factor))

    # Prioritize enemies that are moving toward us
    dot_product = jnp.sum(direction * enemy_velocity, axis=-1)  # (B, 32)
    approaching_mask = dot_product > 0.1
    movement_intensity = movement_intensity * (1.0 + approaching_mask * 0.2)

    # Movement vector
    dvx = direction[:, :, 0] * movement_intensity * 0.2
    dvy = direction[:, :, 1] * movement_intensity * 0.2

    # Flocking behavior: repel from close allies
    ally_positions_expanded = ally_positions[:, :, None, :]  # (B, 32, 1, 2)
    ally_positions_repeated = ally_positions[:, None, :, :]  # (B, 1, 32, 2)
    ally_distances = jnp.linalg.norm(ally_positions_expanded - ally_positions_repeated, axis=-1)  # (B, 32, 32)

    # Repel from close allies
    close_allies_mask = ally_distances < 0.2
    close_allies_mask = jnp.triu(close_allies_mask, k=1)  # Only repel from other allies
    repel_direction = ally_positions_repeated - ally_positions_expanded  # (B, 32, 32, 2)
    repel_direction = jnp.where(
        close_allies_mask[..., None],
        repel_direction,
        jnp.zeros_like(repel_direction)
    )
    repel_force = jnp.sum(repel_direction, axis=2)  # (B, 32, 2)
    repel_norm = jnp.linalg.norm(repel_force, axis=-1, keepdims=True) + 1e-6
    repel_force = repel_force / repel_norm
    dvx += repel_force[:, :, 0] * 0.05
    dvy += repel_force[:, :, 1] * 0.05

    # Attract to the center of the formation
    formation_center = jnp.mean(ally_positions, axis=1)  # (B, 2)
    formation_center_expanded = jnp.broadcast_to(formation_center[:, None, :], (batch_size, num_agents, 2))
    formation_direction = formation_center_expanded - ally_positions
    formation_norm = jnp.linalg.norm(formation_direction, axis=-1, keepdims=True) + 1e-6
    formation_direction = formation_direction / formation_norm
    dvx += formation_direction[:, :, 0] * 0.02
    dvy += formation_direction[:, :, 1] * 0.02

    # Add randomness for exploration
    noise = jrandom.uniform(key, shape=(batch_size, num_agents, 2), minval=-0.05, maxval=0.05)
    dvx += noise[:, :, 0]
    dvy += noise[:, :, 1]

    # Charge behavior: accelerate toward nearest enemy when health is high
    charge_mask = ally_health > 0.9
    dvx = jnp.where(charge_mask, dvx + direction[:, :, 0] * 0.08, dvx)
    dvy = jnp.where(charge_mask, dvy + direction[:, :, 1] * 0.08, dvy)

    # Avoid enemies if health is low
    low_health_mask = ally_health < 0.5
    dvx = jnp.where(low_health_mask, -direction[:, :, 0] * 0.05, dvx)
    dvy = jnp.where(low_health_mask, -direction[:, :, 1] * 0.05, dvy)

    # Avoid wrapping around the map edges
    edge_mask = jnp.logical_or(ally_x < 0.1, ally_x > 0.9)
    edge_mask = jnp.logical_or(edge_mask, jnp.logical_or(ally_y < 0.1, ally_y > 0.9))
    dvx = jnp.where(edge_mask, dvx * 0.5, dvx)
    dvy = jnp.where(edge_mask, dvy * 0.5, dvy)

    # Add formation alignment with velocity matching
    avg_velocity = jnp.mean(ally_velocity, axis=1)  # (B, 2)
    avg_velocity_expanded = jnp.broadcast_to(avg_velocity[:, None, :], (batch_size, num_agents, 2))
    velocity_diff = avg_velocity_expanded - ally_velocity
    velocity_diff_norm = jnp.linalg.norm(velocity_diff, axis=-1, keepdims=True) + 1e-6
    velocity_diff = velocity_diff / velocity_diff_norm
    dvx += velocity_diff[:, :, 0] * 0.01
    dvy += velocity_diff[:, :, 1] * 0.01

    # Avoid enemies when health is very low (additional layer)
    very_low_health_mask = ally_health < 0.2
    dvx = jnp.where(very_low_health_mask, -direction[:, :, 0] * 0.1, dvx)
    dvy = jnp.where(very_low_health_mask, -direction[:, :, 1] * 0.1, dvy)

    # Avoid enemies when health is moderate (additional layer)
    moderate_health_mask = (ally_health < 0.8) & (ally_health > 0.4)
    dvx = jnp.where(moderate_health_mask, -direction[:, :, 0] * 0.03, dvx)
    dvy = jnp.where(moderate_health_mask, -direction[:, :, 1] * 0.03, dvy)

    # Add a strategic behavior: move toward the center of mass of the enemy
    enemy_center = jnp.mean(enemy_positions, axis=1)  # (B, 2)
    enemy_center_expanded = jnp.broadcast_to(enemy_center[:, None, :], (batch_size, num_agents, 2))
    enemy_direction = enemy_center_expanded - ally_positions
    enemy_norm = jnp.linalg.norm(enemy_direction, axis=-1, keepdims=True) + 1e-6
    enemy_direction = enemy_direction / enemy_norm
    dvx += enemy_direction[:, :, 0] * 0.01
    dvy += enemy_direction[:, :, 1] * 0.01

    return dvx, dvy