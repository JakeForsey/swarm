import jax

SPEED = 1

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
    """Agent that rushes the middle."""
    dx = ally_x - 0.5
    dy = ally_y - 0.5
    return dx * SPEED, dy * SPEED
