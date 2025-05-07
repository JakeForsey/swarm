import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Callable
from dataclasses import dataclass
from swarm.env import State

@dataclass
class ConcaveConfig:
    """Configuration for concave swarm behavior"""
    # Formation parameters
    line_length: float = 0.32  # Balanced formation tightness
    curve_strength: float = 0.18  # Moderate curve for stability
    formation_weight: float = 1.1  # Balanced formation priority
    
    # Movement parameters
    advance_speed: float = 0.16  # Balanced advance speed
    retreat_speed: float = 0.19  # Balanced retreat speed
    damping: float = 0.07  # Balanced damping
    
    # Combat parameters
    engagement_distance: float = 0.28  # Balanced engagement range
    min_group_size: int = 2  # Moderate aggression
    health_aggression_scale: float = 1.35  # Balanced aggression scaling

def create_concave_agent(config: ConcaveConfig) -> Callable:
    """Create a concave swarm agent that maintains a curved line formation."""
    def act(state: State, team: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Get team positions and velocities
        if team == 1:
            ally_x, ally_y = state.x1, state.y1
            ally_vx, ally_vy = state.vx1, state.vy1
            ally_health = state.health1
            enemy_x, enemy_y = state.x2, state.y2
        else:
            ally_x, ally_y = state.x2, state.y2
            ally_vx, ally_vy = state.vx2, state.vy2
            ally_health = state.health2
            enemy_x, enemy_y = state.x1, state.y1

        batch_size = ally_x.shape[0]
        num_agents = ally_x.shape[1]
        
        # Shape assertions for input arrays
        assert ally_x.shape == (batch_size, num_agents), f"ally_x shape: {ally_x.shape}, expected ({batch_size}, {num_agents})"
        assert ally_y.shape == (batch_size, num_agents), f"ally_y shape: {ally_y.shape}, expected ({batch_size}, {num_agents})"
        assert ally_vx.shape == (batch_size, num_agents), f"ally_vx shape: {ally_vx.shape}, expected ({batch_size}, {num_agents})"
        assert ally_vy.shape == (batch_size, num_agents), f"ally_vy shape: {ally_vy.shape}, expected ({batch_size}, {num_agents})"
        assert ally_health.shape == (batch_size, num_agents), f"ally_health shape: {ally_health.shape}, expected ({batch_size}, {num_agents})"
        assert enemy_x.shape == (batch_size, num_agents), f"enemy_x shape: {enemy_x.shape}, expected ({batch_size}, {num_agents})"
        assert enemy_y.shape == (batch_size, num_agents), f"enemy_y shape: {enemy_y.shape}, expected ({batch_size}, {num_agents})"

        # Calculate centers with explicit shapes
        ally_com_x = jnp.mean(ally_x, axis=1, keepdims=True)   # Shape: (batch_size, 1)
        ally_com_y = jnp.mean(ally_y, axis=1, keepdims=True)   # Shape: (batch_size, 1)
        enemy_com_x = jnp.mean(enemy_x, axis=1, keepdims=True) # Shape: (batch_size, 1)
        enemy_com_y = jnp.mean(enemy_y, axis=1, keepdims=True) # Shape: (batch_size, 1)
        
        # Shape assertions for center of mass
        assert ally_com_x.shape == (batch_size, 1), f"ally_com_x shape: {ally_com_x.shape}, expected ({batch_size}, 1)"
        assert ally_com_y.shape == (batch_size, 1), f"ally_com_y shape: {ally_com_y.shape}, expected ({batch_size}, 1)"
        assert enemy_com_x.shape == (batch_size, 1), f"enemy_com_x shape: {enemy_com_x.shape}, expected ({batch_size}, 1)"
        assert enemy_com_y.shape == (batch_size, 1), f"enemy_com_y shape: {enemy_com_y.shape}, expected ({batch_size}, 1)"

        # Calculate direction to enemy with explicit shapes
        dx = enemy_com_x - ally_com_x  # Shape: (batch_size, 1)
        dy = enemy_com_y - ally_com_y  # Shape: (batch_size, 1)
        
        # Shape assertions for direction vectors
        assert dx.shape == (batch_size, 1), f"dx shape: {dx.shape}, expected ({batch_size}, 1)"
        assert dy.shape == (batch_size, 1), f"dy shape: {dy.shape}, expected ({batch_size}, 1)"
        
        # Handle wrapping
        dx = jnp.where(dx > 0.5, dx - 1.0, dx)
        dx = jnp.where(dx < -0.5, dx + 1.0, dx)
        dy = jnp.where(dy > 0.5, dy - 1.0, dy)
        dy = jnp.where(dy < -0.5, dy + 1.0, dy)
        
        # Normalize direction
        dist = jnp.sqrt(dx**2 + dy**2) + 1e-5
        dx = dx / dist  # Shape: (batch_size, 1)
        dy = dy / dist  # Shape: (batch_size, 1)
        
        # Verify shapes after normalization
        assert dx.shape == (batch_size, 1), f"normalized dx shape: {dx.shape}, expected ({batch_size}, 1)"
        assert dy.shape == (batch_size, 1), f"normalized dy shape: {dy.shape}, expected ({batch_size}, 1)"

        # Create line positions
        t = jnp.linspace(-1.0, 1.0, num_agents)  # Shape: (num_agents,)
        assert t.shape == (num_agents,), f"t shape before tile: {t.shape}, expected ({num_agents},)"
        
        t = jnp.tile(t[None, :], (batch_size, 1))  # Shape: (batch_size, num_agents)
        assert t.shape == (batch_size, num_agents), f"t shape after tile: {t.shape}, expected ({batch_size}, {num_agents})"

        # Calculate perpendicular direction with explicit reshaping
        perp_x = -dy  # Shape should be (batch_size, 1)
        perp_y = dx   # Shape should be (batch_size, 1)
        
        # Force reshape to ensure (batch_size, 1)
        perp_x = jnp.reshape(perp_x, (batch_size, 1))
        perp_y = jnp.reshape(perp_y, (batch_size, 1))
        
        # Verify perpendicular vector shapes
        assert perp_x.shape == (batch_size, 1), f"perp_x shape: {perp_x.shape}, expected ({batch_size}, 1)"
        assert perp_y.shape == (batch_size, 1), f"perp_y shape: {perp_y.shape}, expected ({batch_size}, 1)"
        
        # Time-based formation scaling with explicit shape
        formation_scale = jnp.minimum(1.0, 1.0 + state.t * 0.1)
        # Reshape formation_scale to (batch_size, 1) for broadcasting
        formation_scale = jnp.reshape(formation_scale, (batch_size, 1))
        assert formation_scale.shape == (batch_size, 1), f"formation_scale shape: {formation_scale.shape}, expected ({batch_size}, 1)"
        
        # Calculate base line positions with explicit broadcasting
        line_x = jnp.multiply(t, perp_x)  # First multiply
        line_x = jnp.multiply(line_x, config.line_length)  # Second multiply
        line_x = jnp.multiply(line_x, formation_scale)  # Third multiply
        
        line_y = jnp.multiply(t, perp_y)
        line_y = jnp.multiply(line_y, config.line_length)
        line_y = jnp.multiply(line_y, formation_scale)
        
        assert line_x.shape == (batch_size, num_agents), f"line_x shape: {line_x.shape}, expected ({batch_size}, {num_agents})"
        assert line_y.shape == (batch_size, num_agents), f"line_y shape: {line_y.shape}, expected ({batch_size}, {num_agents})"
        
        # Add curve toward enemy
        curve = config.curve_strength * (1.0 - t**2)  # Shape: (batch_size, num_agents)
        
        # Ensure dx and dy are properly shaped for broadcasting
        dx_broad = jnp.broadcast_to(dx, (batch_size, num_agents))
        dy_broad = jnp.broadcast_to(dy, (batch_size, num_agents))
        
        curve_x = jnp.multiply(curve, dx_broad)  # Explicit multiply
        curve_x = jnp.multiply(curve_x, config.line_length)
        curve_x = jnp.multiply(curve_x, formation_scale)
        
        curve_y = jnp.multiply(curve, dy_broad)
        curve_y = jnp.multiply(curve_y, config.line_length)
        curve_y = jnp.multiply(curve_y, formation_scale)
        
        assert curve.shape == (batch_size, num_agents), f"curve shape: {curve.shape}, expected ({batch_size}, {num_agents})"
        assert curve_x.shape == (batch_size, num_agents), f"curve_x shape: {curve_x.shape}, expected ({batch_size}, {num_agents})"
        assert curve_y.shape == (batch_size, num_agents), f"curve_y shape: {curve_y.shape}, expected ({batch_size}, {num_agents})"
        
        # Combine for final target positions
        target_x = ally_com_x + line_x + curve_x  # Shape: (batch_size, num_agents)
        target_y = ally_com_y + line_y + curve_y  # Shape: (batch_size, num_agents)
        
        assert target_x.shape == (batch_size, num_agents), f"target_x shape: {target_x.shape}"
        assert target_y.shape == (batch_size, num_agents), f"target_y shape: {target_y.shape}"
        
        # Calculate formation forces
        formation_dx = (target_x - ally_x) * config.formation_weight
        formation_dy = (target_y - ally_y) * config.formation_weight
        
        assert formation_dx.shape == (batch_size, num_agents), f"formation_dx shape: {formation_dx.shape}"
        assert formation_dy.shape == (batch_size, num_agents), f"formation_dy shape: {formation_dy.shape}"
        
        # Calculate combat distances
        enemy_dx = ally_x[:, None, :] - enemy_x[:, :, None]  # Shape: (batch_size, num_enemies, num_agents)
        enemy_dy = ally_y[:, None, :] - enemy_y[:, :, None]
        
        # Handle wrapping for combat
        enemy_dx = jnp.where(enemy_dx > 0.5, enemy_dx - 1.0, enemy_dx)
        enemy_dx = jnp.where(enemy_dx < -0.5, enemy_dx + 1.0, enemy_dx)
        enemy_dy = jnp.where(enemy_dy > 0.5, enemy_dy - 1.0, enemy_dy)
        enemy_dy = jnp.where(enemy_dy < -0.5, enemy_dy + 1.0, enemy_dy)
        
        enemy_dist = jnp.sqrt(enemy_dx**2 + enemy_dy**2)
        
        # Count nearby enemies
        nearby_enemies = jnp.sum(enemy_dist < config.engagement_distance, axis=1)
        
        # Calculate movement speed
        should_advance = nearby_enemies >= config.min_group_size
        move_speed = jnp.where(should_advance, 
                             config.advance_speed * ally_health * config.health_aggression_scale,
                             -config.retreat_speed)
        
        # Add combat movement (ensure proper shapes for broadcasting)
        combat_dx = dx.reshape(batch_size, 1) * move_speed
        combat_dy = dy.reshape(batch_size, 1) * move_speed
        
        assert combat_dx.shape == (batch_size, num_agents), f"combat_dx shape: {combat_dx.shape}"
        assert combat_dy.shape == (batch_size, num_agents), f"combat_dy shape: {combat_dy.shape}"
        
        # Combine forces
        dx = formation_dx + 0.3 * combat_dx
        dy = formation_dy + 0.3 * combat_dy
        
        # Apply damping
        dx = dx - ally_vx * config.damping
        dy = dy - ally_vy * config.damping
        
        # Final shape assertions
        assert dx.shape == (batch_size, num_agents), f"final dx shape: {dx.shape}"
        assert dy.shape == (batch_size, num_agents), f"final dy shape: {dy.shape}"
        
        return dx, dy
    
    return act

# Default configuration
DEFAULT_CONFIG = ConcaveConfig()

# Create default agent
default_agent = create_concave_agent(DEFAULT_CONFIG)

def act(state: State, team: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Main act function that uses the default configuration."""
    return default_agent(state, team, key)
