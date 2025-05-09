import jax
import jax.numpy as jnp

MIN_ACTION = -0.005
MAX_ACTION = 0.005

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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Random agent that moves in completely random directions."""
    xkey, ykey, _ = jax.random.split(key, 3)
    batch_size, num_agents = ally_x.shape
    
    x_action = jax.random.uniform(xkey, (batch_size, num_agents), minval=MIN_ACTION, maxval=MAX_ACTION)
    y_action = jax.random.uniform(ykey, (batch_size, num_agents), minval=MIN_ACTION, maxval=MAX_ACTION)
    
    return x_action, y_action
