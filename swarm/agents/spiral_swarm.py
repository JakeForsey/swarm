import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Callable
from dataclasses import dataclass
from swarm.env import State

@dataclass
class SpiralConfig:
    """Configuration for spiral swarm behavior"""
    # Formation parameters
    base_radius: float = 0.12  # Slightly smaller radius for tighter formation
    rotation_speed: float = 0.3  # Faster rotation
    spiral_tightness: float = 0.2  # Less tight spiral for better combat
    formation_weight: float = 0.08  # Increased formation weight
    velocity_weight: float = 0.1  # Increased velocity matching
    
    # Combat parameters
    chase_radius: float = 0.4  # Larger chase radius
    chase_weight: float = 0.015  # More aggressive chasing
    min_group_size: int = 1  # Allow individual attacks
    health_aggression_scale: float = 1.0  # More aggressive based on health
    perception_radius: float = 0.35  # Larger perception radius
    
    # Movement parameters
    damping: float = 0.08  # Less damping for more responsiveness
    approach_speed: float = 0.15  # Faster approach

def create_spiral_agent(config: SpiralConfig) -> Callable:
    """Create a spiral swarm agent with dynamic center and combat behavior."""
    def act(state: State, team: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Get team positions and velocities
        if team == 1:
            ally_x = state.x1
            ally_y = state.y1
            ally_vx = state.vx1
            ally_vy = state.vy1
            ally_health = state.health1
            enemy_x = state.x2
            enemy_y = state.y2
        else:
            ally_x = state.x2
            ally_y = state.y2
            ally_vx = state.vx2
            ally_vy = state.vy2
            ally_health = state.health2
            enemy_x = state.x1
            enemy_y = state.y1

        # Calculate center of mass for both teams
        ally_com_x = jnp.mean(ally_x, axis=1, keepdims=True)
        ally_com_y = jnp.mean(ally_y, axis=1, keepdims=True)
        enemy_com_x = jnp.mean(enemy_x, axis=1, keepdims=True)
        enemy_com_y = jnp.mean(enemy_y, axis=1, keepdims=True)

        # Calculate direction to enemy center
        dx_to_enemy = enemy_com_x - ally_com_x
        dy_to_enemy = enemy_com_y - ally_com_y
        dist_to_enemy = jnp.sqrt(dx_to_enemy**2 + dy_to_enemy**2)
        
        # Handle wrapping for enemy direction
        dx_to_enemy = jnp.where(dx_to_enemy > 0.5, dx_to_enemy - 1.0, dx_to_enemy)
        dx_to_enemy = jnp.where(dx_to_enemy < -0.5, dx_to_enemy + 1.0, dx_to_enemy)
        dy_to_enemy = jnp.where(dy_to_enemy > 0.5, dy_to_enemy - 1.0, dy_to_enemy)
        dy_to_enemy = jnp.where(dy_to_enemy < -0.5, dy_to_enemy + 1.0, dy_to_enemy)
        
        # Normalize direction to enemy
        dx_to_enemy = dx_to_enemy / (dist_to_enemy + 1e-5)
        dy_to_enemy = dy_to_enemy / (dist_to_enemy + 1e-5)

        # Calculate relative positions to center
        rel_x = ally_x - ally_com_x
        rel_y = ally_y - ally_com_y
        
        # Handle wrapping for relative positions
        rel_x = jnp.where(rel_x > 0.5, rel_x - 1.0, rel_x)
        rel_x = jnp.where(rel_x < -0.5, rel_x + 1.0, rel_x)
        rel_y = jnp.where(rel_y > 0.5, rel_y - 1.0, rel_y)
        rel_y = jnp.where(rel_y < -0.5, rel_y + 1.0, rel_y)
        
        rel_dist = jnp.sqrt(rel_x**2 + rel_y**2)
        
        # Calculate angles for spiral formation
        angles = jnp.arctan2(rel_y, rel_x)
        
        # Calculate target positions in spiral
        # Spiral tightness increases with distance from center
        spiral_factor = 1.0 + config.spiral_tightness * (rel_dist / config.base_radius)
        # Ensure proper broadcasting by reshaping time
        time_factor = jnp.reshape(state.t, (-1, 1)) * config.rotation_speed
        target_angles = angles + time_factor * spiral_factor
        
        # Calculate target positions
        target_x = ally_com_x + config.base_radius * jnp.cos(target_angles)
        target_y = ally_com_y + config.base_radius * jnp.sin(target_angles)
        
        # Add movement toward enemy
        target_x += dx_to_enemy * config.approach_speed
        target_y += dy_to_enemy * config.approach_speed
        
        # Calculate target velocities (tangential to the spiral)
        target_vx = -config.base_radius * jnp.sin(target_angles) * config.rotation_speed
        target_vy = config.base_radius * jnp.cos(target_angles) * config.rotation_speed
        
        # Calculate formation movement
        formation_dx = (target_x - ally_x) * config.formation_weight
        formation_dy = (target_y - ally_y) * config.formation_weight
        
        # Calculate velocity matching
        velocity_match_x = (target_vx - ally_vx) * config.velocity_weight
        velocity_match_y = (target_vy - ally_vy) * config.velocity_weight
        
        # Initialize combat forces
        combat_dx = jnp.zeros_like(ally_x)
        combat_dy = jnp.zeros_like(ally_y)
        
        # Calculate distances to enemies
        enemy_dx = ally_x[:, None, :] - enemy_x[:, :, None]
        enemy_dy = ally_y[:, None, :] - enemy_y[:, :, None]
        
        # Handle wrapping for enemy distances
        enemy_dx = jnp.where(enemy_dx > 0.5, enemy_dx - 1.0, enemy_dx)
        enemy_dx = jnp.where(enemy_dx < -0.5, enemy_dx + 1.0, enemy_dx)
        enemy_dy = jnp.where(enemy_dy > 0.5, enemy_dy - 1.0, enemy_dy)
        enemy_dy = jnp.where(enemy_dy < -0.5, enemy_dy + 1.0, enemy_dy)
        
        enemy_dist = jnp.sqrt(enemy_dx**2 + enemy_dy**2)
        
        # Find closest enemy for each agent
        min_enemy_dist = jnp.min(enemy_dist, axis=1)
        closest_enemy_idx = jnp.argmin(enemy_dist, axis=1)
        
        # Get relative positions to closest enemies
        batch_idx = jnp.arange(ally_x.shape[0])[:, None]
        enemy_idx = closest_enemy_idx
        agent_idx = jnp.arange(ally_x.shape[1])[None, :]
        
        closest_enemy_dx = enemy_dx[batch_idx, enemy_idx, agent_idx]
        closest_enemy_dy = enemy_dy[batch_idx, enemy_idx, agent_idx]
        
        # Calculate aggression based on health and group size
        health_aggression = ally_health * config.health_aggression_scale
        group_size = jnp.sum(enemy_dist < config.perception_radius, axis=1)
        group_advantage = group_size > config.min_group_size
        
        # Chase if we have group advantage and enemy is within range
        chase_mask = (min_enemy_dist < config.chase_radius) & group_advantage
        chase_strength = chase_mask * health_aggression
        
        # Add combat movement
        combat_dx = -closest_enemy_dx * chase_strength * config.chase_weight
        combat_dy = -closest_enemy_dy * chase_strength * config.chase_weight
        
        # Combine all forces
        dx = formation_dx + velocity_match_x + combat_dx
        dy = formation_dy + velocity_match_y + combat_dy
        
        # Apply damping
        dx -= ally_vx * config.damping
        dy -= ally_vy * config.damping
        
        return dx, dy
    
    return act

# Default configuration
DEFAULT_CONFIG = SpiralConfig(
    base_radius=0.18795883994808754,
    rotation_speed=0.27576399081988195,
    spiral_tightness=0.10301011198249985,
    formation_weight=0.141510346628356,
    velocity_weight=0.05460453492299239,
    chase_radius=0.357469506800012,
    chase_weight=0.016298297483683093,
    min_group_size=1,
    health_aggression_scale=1.0762224852357951,
    perception_radius=0.36716339620958827,
    damping=0.10547655251076879,
    approach_speed=0.1658056010939894
)

# Create default agent
default_agent = create_spiral_agent(DEFAULT_CONFIG)

def act(state: State, team: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Main act function that uses the default configuration."""
    return default_agent(state, team, key)
