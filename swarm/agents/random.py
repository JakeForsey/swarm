import jax
import jax.numpy as jnp

from swarm.env import State


MIN_ACTION = -0.005
MAX_ACTION = 0.005


@jax.jit
def act(state: State, team: int, key: jax.random.PRNGKey) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Random agent that moves in completely random directions.
    
    Strategy:
    1. Generates random actions for each agent
    2. Uses uniform distribution for x and y movements
    3. No formation, chasing, or coordinated behavior
    4. Pure random movement for baseline comparison
    
    Parameters:
        state: Current game state containing positions, velocities, and health
        team: Team identifier (1 or 2)
        key: Random key for generating random movements
    
    Returns:
        Tuple of x and y actions for each agent
    """
    xkey, ykey, _ = jax.random.split(key, 3)
    batch_size, num_agents = state.x1.shape
    
    x_action = jax.random.uniform(xkey, (batch_size, num_agents), minval=MIN_ACTION, maxval=MAX_ACTION)
    y_action = jax.random.uniform(ykey, (batch_size, num_agents), minval=MIN_ACTION, maxval=MAX_ACTION)
    
    return x_action, y_action
