from typing import NamedTuple

import jax
import jax.numpy as jnp


class State(NamedTuple):
    # Time
    t: jnp.ndarray

    # Posititions
    x1: jnp.ndarray
    x2: jnp.ndarray
    y1: jnp.ndarray
    y2: jnp.ndarray

    # Velocities
    vx1: jnp.ndarray
    vx2: jnp.ndarray
    vy1: jnp.ndarray
    vy2: jnp.ndarray

    # Health
    health1: jnp.ndarray
    health2: jnp.ndarray

    @property
    def batch_size(self) -> int:
        return self.x1.shape[0]

    @property
    def num_agents(self) -> int:
        return self.x1.shape[1]


class SwarmEnv:
    def __init__(
            self,
            num_agents: int = 32,
            episode_length: int = 128,
            batch_size: int = 1024 * 32,
            max_speed: float = 0.01,
            collision_radius: float = 0.04,
            key: jnp.ndarray = jax.random.PRNGKey(0)
    ):
        self.num_agents = num_agents
        self.episode_length = episode_length
        self.batch_size = batch_size
        self.max_speed = max_speed
        self.collision_radius = collision_radius
        self.key = key

    def reset(self) -> State:
        k1, k2, k3, k4, k5, k6, k7, k8, self.key = jax.random.split(self.key, 9)

        # Initialise time
        t = jnp.zeros((self.batch_size, ))

        shape = (self.batch_size, self.num_agents)
        # Initialize positions
        x1 = jax.random.uniform(k1, shape, minval=0.0, maxval=0.5)
        y1 = jax.random.uniform(k2, shape, minval=0.0, maxval=1.0)
        x2 = jax.random.uniform(k3, shape, minval=0.5, maxval=1.0)
        y2 = jax.random.uniform(k4, shape, minval=0.0, maxval=1.0)
        
        # Initialize velocities
        vx1 = jax.random.uniform(k5, shape, minval=-self.max_speed, maxval=self.max_speed)
        vy1 = jax.random.uniform(k6, shape, minval=-self.max_speed, maxval=self.max_speed)
        vx2 = jax.random.uniform(k7, shape, minval=-self.max_speed, maxval=self.max_speed)
        vy2 = jax.random.uniform(k8, shape, minval=-self.max_speed, maxval=self.max_speed)

        # Initialize health
        health1 = jnp.ones(shape)
        health2 = jnp.ones(shape)
        
        return State(
            t=t,
            x1=x1, x2=x2, y1=y1, y2=y2,
            vx1=vx1, vy1=vy1, vx2=vx2, vy2=vy2,
            health1=health1, health2=health2,
        )
    
    def step(
            self,
            state: State,
            x_action1: jnp.ndarray,
            y_action1: jnp.ndarray,
            x_action2: jnp.ndarray,
            y_action2: jnp.ndarray,
        ) -> tuple[State, jnp.ndarray]:
        return _step(
            state,
            x_action1, y_action1,
            x_action2, y_action2,
            self.episode_length,
            self.max_speed,
            self.collision_radius,
        )


@jax.jit
def _step(
    state: State,
    x_action1: jnp.ndarray,
    y_action1: jnp.ndarray,
    x_action2: jnp.ndarray,
    y_action2: jnp.ndarray,
    episode_length: int,
    max_speed: float,
    collision_radius: float,
) -> tuple[State, jnp.ndarray]:
    # Assert that actions are correct shape and dtype
    action_shape = (state.batch_size, state.num_agents)
    assert x_action1.shape == action_shape
    assert y_action1.shape == action_shape
    assert x_action2.shape == action_shape
    assert y_action2.shape == action_shape
    
    # Update time
    t = state.t + 1

    # Update velocities
    vx1 = jnp.clip(state.vx1 + x_action1, -max_speed, max_speed)
    vx2 = jnp.clip(state.vx2 + x_action2, -max_speed, max_speed)
    vy1 = jnp.clip(state.vy1 + y_action1, -max_speed, max_speed)
    vy2 = jnp.clip(state.vy2 + y_action2, -max_speed, max_speed)

    # Update positions
    x1 = state.x1 + vx1
    x2 = state.x2 + vx2
    y1 = state.y1 + vy1
    y2 = state.y2 + vy2

    # Wrap positions
    x1 = jnp.mod(x1, 1)
    x2 = jnp.mod(x2, 1)
    y1 = jnp.mod(y1, 1)
    y2 = jnp.mod(y2, 1)

    # COLLISIONS
    # Calculate relative positions and velocities for collision detection
    orig_dx = state.x1[:, None, :] - state.x2[:, :, None]
    orig_dy = state.y1[:, None, :] - state.y2[:, :, None]
    
    # Calculate relative velocities
    rel_vx = vx1[:, None, :] - vx2[:, :, None]
    rel_vy = vy1[:, None, :] - vy2[:, :, None]
    
    # Calculate time to closest approach for each pair
    # This is the time when the relative distance is minimized
    rel_v_squared = rel_vx**2 + rel_vy**2
    time_to_closest = - (orig_dx * rel_vx + orig_dy * rel_vy) / (rel_v_squared + 1e-6)
    
    # Clamp time to closest approach to be between 0 and 1 (current timestep)
    time_to_closest = jnp.clip(time_to_closest, 0, 1)
    
    # Calculate minimum distance during the timestep
    min_dx = orig_dx + rel_vx * time_to_closest
    min_dy = orig_dy + rel_vy * time_to_closest
    min_dist = jnp.sqrt(min_dx ** 2 + min_dy ** 2)
    
    # Check for collisions
    collisions = min_dist < collision_radius
    # Mask collisions with dead agents
    alive = (state.health1 > 0)[:, None, :] & (state.health2 > 0)[:, :, None]
    collisions = collisions * alive

    # Update health
    # Each collision reduces health by 0.1
    health1 = state.health1 - jnp.sum(collisions, axis=1) * 0.1
    health2 = state.health2 - jnp.sum(collisions, axis=2) * 0.1
    
    # Clamp health to be between 0 and 1
    health1 = jnp.clip(health1, 0, 1)
    health2 = jnp.clip(health2, 0, 1)

    # Regenerate a small amount of health
    alive1 = health1 > 0
    alive2 = health2 > 0
    health1 += alive1 * 0.01
    health2 += alive2 * 0.01

    # Update reward
    reward = jnp.where(
        # If the game is finished
        t == episode_length,
        # Assign 1 if team 1 has more health, -1 if team 2 has more health
        jnp.where(health1.sum(axis=1) > health2.sum(axis=1), 1, -1),
        # If the health is identical, assign 0
        0,
    )

    return (
        State(
            t=t,
            x1=x1, x2=x2, y1=y1, y2=y2,
            vx1=vx1, vx2=vx2, vy1=vy1, vy2=vy2,
            health1=health1, health2=health2,
        ),
        reward,
    )
