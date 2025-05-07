import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Callable
from dataclasses import dataclass
from swarm.env import State

@dataclass
class FormationConfig:
    """Configuration for formation behavior"""
    scale: float = 0.7  # How tightly agents cluster (0 = spread, 1 = tight)
    shape: float = 0.3  # Shape of formation (0 = circle, 1 = line)
    weight: float = 1.0  # How much to prioritize formation

@dataclass
class CombatConfig:
    """Configuration for combat behavior"""
    aggressiveness: float = 0.6  # How aggressively to approach enemies
    attack_threshold: float = 0.3  # Health threshold to start attacking
    retreat_threshold: float = 0.2  # Health threshold to start retreating
    weight: float = 1.0  # How much to prioritize combat

@dataclass
class MovementConfig:
    """Configuration for movement behavior"""
    max_speed: float = 0.01
    smoothness: float = 0.8  # How smoothly to change direction
    damping: float = 0.1  # Velocity damping factor

@dataclass
class SwarmConfig:
    """Complete configuration for the swarm agent"""
    formation: FormationConfig = FormationConfig()
    combat: CombatConfig = CombatConfig()
    movement: MovementConfig = MovementConfig()

def compute_formation_forces(
    ally_x: jnp.ndarray,
    ally_y: jnp.ndarray,
    config: FormationConfig
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute forces to maintain formation."""
    batch_size, num_agents = ally_x.shape
    
    # Compute center of mass
    com_x = jnp.mean(ally_x, axis=1, keepdims=True)
    com_y = jnp.mean(ally_y, axis=1, keepdims=True)
    
    # Compute relative positions
    rel_x = ally_x - com_x
    rel_y = ally_y - com_y
    
    # Compute formation target positions
    angles = jnp.linspace(0, 2 * jnp.pi, num_agents, endpoint=False)
    circle_x = jnp.cos(angles) * config.scale
    circle_y = jnp.sin(angles) * config.scale
    
    line_x = jnp.linspace(-config.scale, config.scale, num_agents)
    line_y = jnp.zeros_like(line_x)
    
    # Blend between circle and line
    target_x = (1 - config.shape) * circle_x + config.shape * line_x
    target_y = (1 - config.shape) * circle_y + config.shape * line_y
    
    # Compute formation forces
    dx = target_x - rel_x
    dy = target_y - rel_y
    
    # Normalize and scale by weight
    force_mag = jnp.sqrt(dx**2 + dy**2)
    dx = dx / (force_mag + 1e-5) * config.weight
    dy = dy / (force_mag + 1e-5) * config.weight
    
    return dx, dy

def compute_combat_forces(
    ally_x: jnp.ndarray,
    ally_y: jnp.ndarray,
    ally_health: jnp.ndarray,
    enemy_x: jnp.ndarray,
    enemy_y: jnp.ndarray,
    config: CombatConfig
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute forces for combat behavior."""
    batch_size, num_agents = ally_x.shape
    
    # Find nearest enemy for each agent
    dx = enemy_x[:, None, :] - ally_x[:, :, None]
    dy = enemy_y[:, None, :] - ally_y[:, :, None]
    distances = jnp.sqrt(dx**2 + dy**2)
    nearest_enemy_idx = jnp.argmin(distances, axis=-1)
    
    # Get nearest enemy positions
    batch_indices = jnp.arange(batch_size)[:, None]
    nearest_enemy_x = enemy_x[batch_indices, nearest_enemy_idx]
    nearest_enemy_y = enemy_y[batch_indices, nearest_enemy_idx]
    
    # Compute enemy forces
    enemy_dx = nearest_enemy_x - ally_x
    enemy_dy = nearest_enemy_y - ally_y
    enemy_dist = jnp.sqrt(enemy_dx**2 + enemy_dy**2)
    
    # Normalize enemy forces
    enemy_dx = enemy_dx / (enemy_dist + 1e-5)
    enemy_dy = enemy_dy / (enemy_dist + 1e-5)
    
    # Determine attack/retreat behavior based on health
    health_factor = (ally_health - config.retreat_threshold) / (config.attack_threshold - config.retreat_threshold)
    health_factor = jnp.clip(health_factor, 0, 1)
    
    # Scale by aggressiveness and weight
    enemy_dx = enemy_dx * health_factor * config.aggressiveness * config.weight
    enemy_dy = enemy_dy * health_factor * config.aggressiveness * config.weight
    
    return enemy_dx, enemy_dy

def compute_movement(
    dx: jnp.ndarray,
    dy: jnp.ndarray,
    ally_vx: jnp.ndarray,
    ally_vy: jnp.ndarray,
    config: MovementConfig
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute final movement actions."""
    # Normalize combined forces
    force_mag = jnp.sqrt(dx**2 + dy**2)
    dx = dx / (force_mag + 1e-5)
    dy = dy / (force_mag + 1e-5)
    
    # Scale by max speed
    x_action = dx * config.max_speed
    y_action = dy * config.max_speed
    
    # Apply smoothness to velocity changes
    x_action = config.smoothness * x_action + (1 - config.smoothness) * ally_vx
    y_action = config.smoothness * y_action + (1 - config.smoothness) * ally_vy
    
    # Apply damping
    x_action = x_action - ally_vx * config.damping
    y_action = y_action - ally_vy * config.damping
    
    return x_action, y_action

def create_swarm_agent(config: SwarmConfig) -> Callable:
    """Create a swarm agent with the given configuration."""
    def act(state: State, team: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
        
        # Compute formation forces
        formation_dx, formation_dy = compute_formation_forces(
            ally_x, ally_y, config.formation
        )
        
        # Compute combat forces
        combat_dx, combat_dy = compute_combat_forces(
            ally_x, ally_y, ally_health,
            enemy_x, enemy_y,
            config.combat
        )
        
        # Combine forces
        dx = formation_dx + combat_dx
        dy = formation_dy + combat_dy
        
        # Compute final movement
        return compute_movement(dx, dy, ally_vx, ally_vy, config.movement)
    
    return act

# Best configuration from optimization
BEST_CONFIG = SwarmConfig(
    formation=FormationConfig(
        scale=0.22838001522636964,
        shape=0.385398794029561,
        weight=0.2238010236691683
    ),
    combat=CombatConfig(
        aggressiveness=1.1672251553679267,
        attack_threshold=0.357952049355146,
        retreat_threshold=0.3258188515909997,
        weight=0.8086085161922798
    ),
    movement=MovementConfig(
        max_speed=0.012270015706612979,
        smoothness=0.9792381288997192,
        damping=0.11650352192738242
    )
)

# Create the best agent
best_agent = create_swarm_agent(BEST_CONFIG)

def act(state: State, team: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Main act function that uses the best configuration."""
    return best_agent(state, team, key)
