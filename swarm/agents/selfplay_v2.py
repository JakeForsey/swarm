
import jax
import jax.numpy as jnp
from jax import jit

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

    # Normalize all positions to handle wrap-around
    ally_x = jnp.mod(ally_x, 1.0)
    ally_y = jnp.mod(ally_y, 1.0)
    enemy_x = jnp.mod(enemy_x, 1.0)
    enemy_y = jnp.mod(enemy_y, 1.0)

    # ---FORMATION: Move in a circular formation to avoid clustering---
    # Compute formation center (center of mass of allies)
    ally_center_x = jnp.mean(ally_x, axis=1, keepdims=True)
    ally_center_y = jnp.mean(ally_y, axis=1, keepdims=True)

    # Compute relative position within formation
    rel_x = ally_x - ally_center_x
    rel_y = ally_y - ally_center_y

    # Normalize to get direction from center
    dist_to_center = jnp.sqrt(rel_x**2 + rel_y**2) + 1e-6
    dir_x = rel_x / dist_to_center
    dir_y = rel_y / dist_to_center

    # Move away from center slightly to form a circle
    formation_dir_x = -dir_x * 0.05
    formation_dir_y = -dir_y * 0.05

    # ---TARGET ENEMY: Chase the nearest enemy---
    # Compute pairwise distance between allies and enemies
    dx = enemy_x[:, None, :] - ally_x[:, :, None]  # [B, A, E]
    dy = enemy_y[:, None, :] - ally_y[:, :, None]  # [B, A, E]
    dist = jnp.sqrt(dx**2 + dy**2) + 1e-6  # [B, A, E]

    # Find the index of the nearest enemy for each ally
    nearest_idx = jnp.argmin(dist, axis=2)
    nearest_x = jnp.take_along_axis(enemy_x[:, None, :], nearest_idx[:, :, None], axis=2).squeeze(2)
    nearest_y = jnp.take_along_axis(enemy_y[:, None, :], nearest_idx[:, :, None], axis=2).squeeze(2)

    # Compute direction toward nearest enemy
    target_dir_x = nearest_x - ally_x
    target_dir_y = nearest_y - ally_y
    norm = jnp.sqrt(target_dir_x**2 + target_dir_y**2) + 1e-6
    target_dir_x = target_dir_x / norm
    target_dir_y = target_dir_y / norm

    # Apply acceleration toward the nearest enemy
    pursuit_dir_x = target_dir_x * 0.2
    pursuit_dir_y = target_dir_y * 0.2

    # ---COLLISION AVOIDANCE: Repel from nearby enemies---
    # Repel if distance < 0.2
    repel_mask = dist < 0.2
    repel_dir_x = (ally_x[:, :, None] - enemy_x[:, None, :]) * repel_mask
    repel_dir_y = (ally_y[:, :, None] - enemy_y[:, None, :]) * repel_mask

    # Normalize and sum repulsion from all nearby enemies
    repel_norm = jnp.sqrt(repel_dir_x**2 + repel_dir_y**2) + 1e-6
    repel_dir_x = repel_dir_x / repel_norm
    repel_dir_y = repel_dir_y / repel_norm
    repel_dir_x = jnp.sum(repel_dir_x, axis=2)
    repel_dir_y = jnp.sum(repel_dir_y, axis=2)

    repel_dir_x = repel_dir_x * 0.05
    repel_dir_y = repel_dir_y * 0.05

    # ---HEALTH MANAGEMENT: If low health, move towards formation center---
    low_health_mask = ally_health < 0.5
    formation_dir_x = jnp.where(low_health_mask, formation_dir_x * 0.5, formation_dir_x)
    formation_dir_y = jnp.where(low_health_mask, formation_dir_y * 0.5, formation_dir_y)

    # ---RANDOM NOISE: Add small random movement to avoid symmetry---
    noise = 0.01 * jax.random.normal(key, (batch_size, num_agents))
    formation_dir_x += noise
    formation_dir_y += noise

    # ---Final acceleration vector: formation + pursuit + repulsion---
    dvx = formation_dir_x + pursuit_dir_x + repel_dir_x
    dvy = formation_dir_y + pursuit_dir_y + repel_dir_y

    # Clip to avoid extreme acceleration
    dvx = jnp.clip(dvx, -1.0, 1.0)
    dvy = jnp.clip(dvy, -1.0, 1.0)

    return dvx, dvy
