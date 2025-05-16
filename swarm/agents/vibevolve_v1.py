import jax
import jax.numpy as jnp
from jax import jit, vmap

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
    MOVE_SPEED = 0.7
    AVOID_RADIUS = 1.0
    HEALTH_THRESHOLD = 0.5  # If health is below this, prioritize regenerating
    REGENERATION_RATE = 0.02  # Health regenerates over time
    FORMATION_RADIUS = 5.0  # Radius of the formation circle
    TARGET_RADIUS = 2.0     # Radius to consider an enemy as a target
    AVOIDANCE_STRENGTH = 1.5
    FORMATION_STRENGTH = 0.5
    TARGET_STRENGTH = 2.0
    HEALTH_STRENGTH = 1.0
    MAX_HEALTH = 1.0
    DAMPING = 0.9  # Dampen velocity to avoid overshooting
    SEPARATION_RADIUS = 1.5
    PREDATOR_DISTANCE = 3.0  # Distance to start chasing enemies
    PREDATOR_SPEED = 1.2

    # Compute relative positions to enemies
    ally_to_enemy_dx = enemy_x[:, None, :] - ally_x[:, :, None]  # (B, A, E)
    ally_to_enemy_dy = enemy_y[:, None, :] - ally_y[:, :, None]  # (B, A, E)

    # Compute distance squared to each enemy
    dist_sq = ally_to_enemy_dx**2 + ally_to_enemy_dy**2  # (B, A, E)

    # Find the index of the closest enemy for each agent
    min_dist_idx = jnp.argmin(dist_sq, axis=2)  # (B, A)
    min_dist_dx = jnp.take_along_axis(ally_to_enemy_dx, min_dist_idx[:, :, None], axis=2).squeeze(2)  # (B, A)
    min_dist_dy = jnp.take_along_axis(ally_to_enemy_dy, min_dist_idx[:, :, None], axis=2).squeeze(2)  # (B, A)

    # Compute direction toward the closest enemy
    dist = jnp.sqrt(min_dist_dx**2 + min_dist_dy**2 + 1e-8)
    enemy_direction_x = min_dist_dx / dist
    enemy_direction_y = min_dist_dy / dist

    # Avoid close allies
    ally_to_ally_dx = ally_x[:, :, None] - ally_x[:, None, :]  # (B, A, A)
    ally_to_ally_dy = ally_y[:, :, None] - ally_y[:, None, :]  # (B, A, A)
    ally_dist_sq = ally_to_ally_dx**2 + ally_to_ally_dy**2  # (B, A, A)
    ally_mask = ally_dist_sq < (SEPARATION_RADIUS ** 2)  # (B, A, A)
    ally_mask = jnp.any(ally_mask, axis=2)  # (B, A): True if any ally is too close

    # Avoid close enemies
    enemy_to_ally_dx = ally_x[:, :, None] - enemy_x[:, None, :]  # (B, A, E)
    enemy_to_ally_dy = ally_y[:, :, None] - enemy_y[:, None, :]  # (B, A, E)
    enemy_dist_sq = enemy_to_ally_dx**2 + enemy_to_ally_dy**2  # (B, A, E)
    enemy_mask = enemy_dist_sq < (AVOID_RADIUS ** 2)  # (B, A, E)
    enemy_mask = jnp.any(enemy_mask, axis=2)  # (B, A): True if any enemy is too close

    # Combine all avoidance masks
    avoid_mask = ally_mask | enemy_mask
    avoid_factor = jnp.where(avoid_mask, 0.5, 1.0)  # Reduce movement when avoiding

    # Compute health-based behavior
    low_health = ally_health < HEALTH_THRESHOLD
    health_factor = jnp.where(low_health, 0.0, 1.0)  # If low on health, stop moving

    # Compute formation-based movement: move to a circular formation
    formation_center_x = jnp.mean(ally_x, axis=1, keepdims=True)
    formation_center_y = jnp.mean(ally_y, axis=1, keepdims=True)

    # Compute direction to move to the formation
    formation_dir_x = formation_center_x - ally_x
    formation_dir_y = formation_center_y - ally_y

    # Normalize direction
    formation_dist = jnp.sqrt(formation_dir_x**2 + formation_dir_y**2 + 1e-8)
    formation_dir_x = formation_dir_x / formation_dist
    formation_dir_y = formation_dir_y / formation_dist

    # Compute separation from other allies
    separation_dir_x = jnp.mean(ally_to_ally_dx * (1 - jnp.eye(num_agents)[None, ...]), axis=2)
    separation_dir_y = jnp.mean(ally_to_ally_dy * (1 - jnp.eye(num_agents)[None, ...]), axis=2)
    separation_dist = jnp.sqrt(separation_dir_x**2 + separation_dir_y**2 + 1e-8)
    separation_dir_x = separation_dir_x / (separation_dist + 1e-8)
    separation_dir_y = separation_dir_y / (separation_dist + 1e-8)
    separation_factor = jnp.clip(formation_dist / SEPARATION_RADIUS, 0.0, 1.0)
    separation_dir_x = separation_dir_x * separation_factor
    separation_dir_y = separation_dir_y * separation_factor

    # Compute target direction: enemies within a certain radius are targets
    target_mask = dist < TARGET_RADIUS
    predator_mask = dist < PREDATOR_DISTANCE
    target_direction_x = jnp.where(target_mask, enemy_direction_x, formation_dir_x)
    target_direction_y = jnp.where(target_mask, enemy_direction_y, formation_dir_y)
    predator_direction_x = jnp.where(predator_mask, enemy_direction_x, formation_dir_x)
    predator_direction_y = jnp.where(predator_mask, enemy_direction_y, formation_dir_y)

    # Weighted movement: move toward target with formation fallback
    dvx = target_direction_x * TARGET_STRENGTH * MOVE_SPEED * avoid_factor * health_factor
    dvy = target_direction_y * TARGET_STRENGTH * MOVE_SPEED * avoid_factor * health_factor

    # Add separation component
    dvx += separation_dir_x * AVOIDANCE_STRENGTH * MOVE_SPEED * avoid_factor
    dvy += separation_dir_y * AVOIDANCE_STRENGTH * MOVE_SPEED * avoid_factor

    # Add formation-based movement
    formation_component_x = formation_dir_x * FORMATION_STRENGTH * MOVE_SPEED * avoid_factor * health_factor
    formation_component_y = formation_dir_y * FORMATION_STRENGTH * MOVE_SPEED * avoid_factor * health_factor
    dvx += formation_component_x
    dvy += formation_component_y

    # Add predator behavior: chase enemies within range
    predator_component_x = predator_direction_x * PREDATOR_SPEED * avoid_factor * (1 - health_factor)
    predator_component_y = predator_direction_y * PREDATOR_SPEED * avoid_factor * (1 - health_factor)
    dvx += predator_component_x
    dvy += predator_component_y

    # Regenerate health by moving toward the center of the formation
    regen_dir_x = formation_dir_x
    regen_dir_y = formation_dir_y
    regen_factor = jnp.where(low_health, 1.0, 0.0)  # Only regenerate when low on health
    dvx = dvx * (1 - regen_factor) + regen_dir_x * regen_factor
    dvy = dvy * (1 - regen_factor) + regen_dir_y * regen_factor

    # Apply velocity dampening
    dvx = dvx * DAMPING + ally_vx * (1 - DAMPING)
    dvy = dvy * DAMPING + ally_vy * (1 - DAMPING)

    # Apply regeneration rate to health
    ally_health = ally_health + REGENERATION_RATE * (MAX_HEALTH - ally_health)

    return dvx, dvy