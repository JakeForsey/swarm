import jax
import jax.numpy as jnp
from jax import jit, vmap, random

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
    MAX_HEALTH = 1.0
    REGEN_RATE = 0.02
    MIN_HEALTH_TO_ATTACK = 0.5
    ACCEL_MAG = 0.1
    RANDOM_PERTURBATION = 0.01
    SEPARATION_RADIUS = 0.1
    SEPARATION_FORCE = 0.05
    COHESION_RADIUS = 0.5
    COHESION_FORCE = 0.02
    ATTACK_PRIORITY_WEIGHT = 0.8
    REGEN_PRIORITY_WEIGHT = 0.2
    HEALTH_WEIGHTING_SLOPE = 20.0
    EDGE_BUFFER = 0.05
    TEAM_COHESION_FORCE = 0.02
    MAX_VEL = 0.3
    HEALTH_THRESHOLD = 0.3
    ALIVE_CHECK = 0.1
    LOW_HEALTH_THRESHOLD = 0.25
    ATTACK_RADIUS = 0.25
    TARGETING_RADIUS = 0.3
    FLEE_DISTANCE = 0.5
    TEAM_AWARENESS_RADIUS = 0.8
    EDGE_AVOIDANCE_FORCE = 0.05
    ENEMY_AVOIDANCE_FORCE = 0.07

    # Normalize health
    ally_health_normalized = ally_health / MAX_HEALTH
    ally_alive = ally_health_normalized > ALIVE_CHECK
    enemy_health_normalized = enemy_health / MAX_HEALTH
    enemy_alive = enemy_health_normalized > ALIVE_CHECK

    # Compute distance to enemies
    dx = jnp.expand_dims(ally_x, axis=-1) - jnp.expand_dims(enemy_x, axis=1)
    dy = jnp.expand_dims(ally_y, axis=-1) - jnp.expand_dims(enemy_y, axis=1)
    dist_to_enemies = jnp.sqrt(dx**2 + dy**2 + 1e-6)
    enemy_alive_mask = jnp.expand_dims(enemy_alive, axis=1)
    valid_dist_to_enemies = jnp.where(enemy_alive_mask, dist_to_enemies, jnp.inf)

    # Find closest and furthest enemies to each ally
    closest_enemy_idx = jnp.argmin(valid_dist_to_enemies, axis=-1)
    furthest_enemy_idx = jnp.argmax(valid_dist_to_enemies, axis=-1)

    # Get closest and furthest enemy positions
    closest_enemy_x = jnp.take_along_axis(enemy_x, closest_enemy_idx, axis=-1)
    closest_enemy_y = jnp.take_along_axis(enemy_y, closest_enemy_idx, axis=-1)
    furthest_enemy_x = jnp.take_along_axis(enemy_x, furthest_enemy_idx, axis=-1)
    furthest_enemy_y = jnp.take_along_axis(enemy_y, furthest_enemy_idx, axis=-1)

    # Compute direction to closest enemy
    direction_x_closest = closest_enemy_x - ally_x
    direction_y_closest = closest_enemy_y - ally_y
    direction_mag_closest = jnp.sqrt(direction_x_closest**2 + direction_y_closest**2 + 1e-6)
    direction_x_closest = direction_x_closest / (direction_mag_closest + 1e-6)
    direction_y_closest = direction_y_closest / (direction_mag_closest + 1e-6)
    away_from_closest_x = -direction_x_closest
    away_from_closest_y = -direction_y_closest

    # Compute direction to furthest enemy
    direction_x_furthest = furthest_enemy_x - ally_x
    direction_y_furthest = furthest_enemy_y - ally_y
    direction_mag_furthest = jnp.sqrt(direction_x_furthest**2 + direction_y_furthest**2 + 1e-6)
    direction_x_furthest = direction_x_furthest / (direction_mag_furthest + 1e-6)
    direction_y_furthest = direction_y_furthest / (direction_mag_furthest + 1e-6)

    # Compute separation from other allies
    dx_allies = jnp.expand_dims(ally_x, axis=-1) - jnp.expand_dims(ally_x, axis=1)
    dy_allies = jnp.expand_dims(ally_y, axis=-1) - jnp.expand_dims(ally_y, axis=1)
    dist_to_allies = jnp.sqrt(dx_allies**2 + dy_allies**2 + 1e-6)
    separation_mask = (dist_to_allies < SEPARATION_RADIUS) & (jnp.expand_dims(ally_alive, axis=1) & jnp.expand_dims(ally_alive, axis=1))

    # Apply separation force
    separation_dx = jnp.where(separation_mask, dx_allies, 0.0)
    separation_dy = jnp.where(separation_mask, dy_allies, 0.0)
    separation_mag = jnp.sqrt(separation_dx**2 + separation_dy**2 + 1e-6)
    separation_x = separation_dx / (separation_mag + 1e-6)
    separation_y = separation_dy / (separation_mag + 1e-6)
    separation_x = jnp.sum(separation_x, axis=1)
    separation_y = jnp.sum(separation_y, axis=1)

    # Compute cohesion to center of allies
    team_center_x = jnp.mean(ally_x, axis=1, keepdims=True)
    team_center_y = jnp.mean(ally_y, axis=1, keepdims=True)
    dx_to_center = ally_x - team_center_x
    dy_to_center = ally_y - team_center_y
    dist_to_center = jnp.sqrt(dx_to_center**2 + dy_to_center**2 + 1e-6)
    dist_to_center = jnp.clip(dist_to_center, 0.1, 1.0)
    cohesion_x = dx_to_center / dist_to_center * COHESION_FORCE
    cohesion_y = dy_to_center / dist_to_center * COHESION_FORCE

    # Compute team-aware cohesion
    dx_to_teammates = jnp.expand_dims(ally_x, axis=-1) - jnp.expand_dims(ally_x, axis=1)
    dy_to_teammates = jnp.expand_dims(ally_y, axis=-1) - jnp.expand_dims(ally_y, axis=1)
    dist_to_teammates = jnp.sqrt(dx_to_teammates**2 + dy_to_teammates**2 + 1e-6)
    team_awareness_mask = (dist_to_teammates < TEAM_AWARENESS_RADIUS) & (jnp.expand_dims(ally_alive, axis=1) & jnp.expand_dims(ally_alive, axis=1))
    team_awareness_dx = jnp.where(team_awareness_mask, dx_to_teammates, 0.0)
    team_awareness_dy = jnp.where(team_awareness_mask, dy_to_teammates, 0.0)
    team_awareness_mag = jnp.sqrt(team_awareness_dx**2 + team_awareness_dy**2 + 1e-6)
    team_awareness_x = team_awareness_dx / (team_awareness_mag + 1e-6)
    team_awareness_y = team_awareness_dy / (team_awareness_mag + 1e-6)
    team_awareness_x = jnp.sum(team_awareness_x, axis=1)
    team_awareness_y = jnp.sum(team_awareness_y, axis=1)
    team_awareness_x = team_awareness_x * TEAM_COHESION_FORCE
    team_awareness_y = team_awareness_y * TEAM_COHESION_FORCE

    # Compute attack/health-based weighting
    attack_weight = 1.0 / (1.0 + jnp.exp(-HEALTH_WEIGHTING_SLOPE * (ally_health_normalized - MIN_HEALTH_TO_ATTACK)))
    regenerate_weight = 1.0 - attack_weight

    # Adjust weights based on health level
    health_based_weight = jnp.where(ally_health_normalized < HEALTH_THRESHOLD, 1.0, 0.0)
    attack_weight = jnp.where(health_based_weight > 0.5, 0.2, attack_weight)
    regenerate_weight = jnp.where(health_based_weight > 0.5, 0.8, regenerate_weight)

    # Prioritize enemies with low health
    closest_enemy_health = jnp.take_along_axis(enemy_health_normalized, closest_enemy_idx, axis=-1)
    low_health_attack_weight = jnp.where(closest_enemy_health < LOW_HEALTH_THRESHOLD, 1.5, 1.0)
    attack_weight = attack_weight * low_health_attack_weight

    # Compute attack/avoid direction
    weighted_dir_x = attack_weight * direction_x_closest + regenerate_weight * away_from_closest_x
    weighted_dir_y = attack_weight * direction_y_closest + regenerate_weight * away_from_closest_y

    # Add separation and cohesion forces
    weighted_dir_x += separation_x * SEPARATION_FORCE
    weighted_dir_y += separation_y * SEPARATION_FORCE
    weighted_dir_x += cohesion_x
    weighted_dir_y += cohesion_y
    weighted_dir_x += team_awareness_x
    weighted_dir_y += team_awareness_y

    # Avoid getting too close to map edges
    edge_distance_x = jnp.minimum(ally_x, 1.0 - ally_x)
    edge_distance_y = jnp.minimum(ally_y, 1.0 - ally_y)
    edge_force_x = jnp.where(edge_distance_x < EDGE_BUFFER, -jnp.sign(ally_x - 0.5), 0.0)
    edge_force_y = jnp.where(edge_distance_y < EDGE_BUFFER, -jnp.sign(ally_y - 0.5), 0.0)
    weighted_dir_x += edge_force_x * EDGE_AVOIDANCE_FORCE
    weighted_dir_y += edge_force_y * EDGE_AVOIDANCE_FORCE

    # Compute targeting radius for better engagement
    target_distance = jnp.sqrt(direction_x_closest**2 + direction_y_closest**2 + 1e-6)
    target_dir_x = jnp.where(target_distance < TARGETING_RADIUS, direction_x_closest, 0.0)
    target_dir_y = jnp.where(target_distance < TARGETING_RADIUS, direction_y_closest, 0.0)
    target_dir_x = target_dir_x * attack_weight
    target_dir_y = target_dir_y * attack_weight
    weighted_dir_x += target_dir_x
    weighted_dir_y += target_dir_y

    # Add velocity-based directional bias
    vel_dir_x = ally_vx / (jnp.sqrt(ally_vx**2 + ally_vy**2 + 1e-6) + 1e-6)
    vel_dir_y = ally_vy / (jnp.sqrt(ally_vx**2 + ally_vy**2 + 1e-6) + 1e-6)
    weighted_dir_x += vel_dir_x * 0.02
    weighted_dir_y += vel_dir_y * 0.02

    # Add enemy avoidance to furthest enemy
    enemy_avoidance_x = direction_x_furthest * ENEMY_AVOIDANCE_FORCE
    enemy_avoidance_y = direction_y_furthest * ENEMY_AVOIDANCE_FORCE
    weighted_dir_x += enemy_avoidance_x
    weighted_dir_y += enemy_avoidance_y

    # Apply acceleration with velocity normalization
    dvx = weighted_dir_x * ACCEL_MAG
    dvy = weighted_dir_y * ACCEL_MAG

    # Normalize velocity magnitude to avoid overshooting
    vel_mag = jnp.sqrt(dvx**2 + dvy**2 + 1e-6)
    dvx = dvx * (MAX_VEL / (vel_mag + 1e-6))
    dvy = dvy * (MAX_VEL / (vel_mag + 1e-6))

    # Add random perturbation
    random_key, subkey = random.split(key)
    noise_x = random.uniform(subkey, (batch_size, num_agents)) * RANDOM_PERTURBATION * 2 - RANDOM_PERTURBATION
    noise_y = random.uniform(subkey, (batch_size, num_agents)) * RANDOM_PERTURBATION * 2 - RANDOM_PERTURBATION

    dvx += noise_x
    dvy += noise_y

    return dvx, dvy