import jax
import jax.numpy as jnp

DAMPING = 0.2  # Very strong damping to prevent any movement

@jax.jit
def act(
    t,
    key,
    ally_x,
    ally_y,
    ally_vx,
    ally_vy,
    ally_health,
    enemy_y,
    enemy_x,
    enemy_vx,
    enemy_vy,
    enemy_health,
):
    """Static swarm agent that keeps all agents exactly where they start."""
    # Initialize actions
    x_action = jnp.zeros_like(ally_vx)
    y_action = jnp.zeros_like(ally_vy)
    
    # Apply strong damping to prevent any movement
    x_action -= ally_vx * DAMPING
    y_action -= ally_vy * DAMPING
    
    return x_action, y_action
