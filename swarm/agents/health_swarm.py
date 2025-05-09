import jax
import jax.numpy as jnp
from typing import Tuple

attack_speed: float = 0.2  # Speed when attacking
flee_speed: float = 0.15   # Speed when fleeing

@jax.jit
def act(
    t: jnp.ndarray,
    key: jnp.ndarray,
    ally_x: jnp.ndarray,
    ally_y: jnp.ndarray,
    ally_vx: jnp.ndarray,
    ally_vy: jnp.ndarray,
    ally_health: jnp.ndarray,
    enemy_y: jnp.ndarray,
    enemy_x: jnp.ndarray,
    enemy_vx: jnp.ndarray,
    enemy_vy: jnp.ndarray,
    enemy_health: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    batch_size = ally_x.shape[0]
    num_allies = ally_x.shape[1]
    
    # Calculate distances to all enemies
    # Shape: (batch_size, num_allies, num_enemies)
    dx = ally_x[:, :, None] - enemy_x[:, None, :]
    dy = ally_y[:, :, None] - enemy_y[:, None, :]
    
    # Handle wrapping
    dx = jnp.where(dx > 0.5, dx - 1.0, dx)
    dx = jnp.where(dx < -0.5, dx + 1.0, dx)
    dy = jnp.where(dy > 0.5, dy - 1.0, dy)
    dy = jnp.where(dy < -0.5, dy + 1.0, dy)
    
    # Calculate distances
    distances = jnp.sqrt(dx**2 + dy**2) + 1e-5
    
    # Create mask for alive enemies (health > 0)
    alive_enemies = enemy_health > 0  # Shape: (batch_size, num_enemies)
    
    # Set distance to dead enemies to infinity so they won't be chosen
    distances = jnp.where(alive_enemies[:, None, :], distances, jnp.inf)
    
    # Find nearest enemy for each agent
    nearest_enemy_idx = jnp.argmin(distances, axis=2)  # Shape: (batch_size, num_allies)
    
    # Get health of nearest enemies
    batch_idx = jnp.arange(batch_size)[:, None]
    agent_idx = jnp.arange(num_allies)[None, :]
    nearest_enemy_health = enemy_health[batch_idx, nearest_enemy_idx]
    
    # Compare health with nearest enemy
    health_diff = ally_health - nearest_enemy_health
    should_attack = health_diff >= 0
    # should_attack = health_diff > config.health_threshold
    
    # Get direction to nearest enemy
    nearest_enemy_dx = dx[batch_idx, agent_idx, nearest_enemy_idx]
    nearest_enemy_dy = dy[batch_idx, agent_idx, nearest_enemy_idx]
    
    # Calculate movement based on health comparison
    move_speed = jnp.where(should_attack, -attack_speed, flee_speed)
    dx = nearest_enemy_dx * move_speed
    dy = nearest_enemy_dy * move_speed
            
    return dx, dy
