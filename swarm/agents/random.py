import jax

MIN_ACTION = -0.005
MAX_ACTION = 0.005

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
    """Random agent that moves in completely random directions."""
    xkey, ykey, _ = jax.random.split(key, 3)
    batch_size, num_agents = ally_x.shape
    
    x_action = jax.random.uniform(xkey, (batch_size, num_agents), minval=MIN_ACTION, maxval=MAX_ACTION)
    y_action = jax.random.uniform(ykey, (batch_size, num_agents), minval=MIN_ACTION, maxval=MAX_ACTION)
    
    return x_action, y_action
