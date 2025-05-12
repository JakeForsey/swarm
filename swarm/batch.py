from itertools import combinations

import jax
import jax.numpy as jnp
from swarm.env import State


@jax.jit
def get_indices(mask: jnp.ndarray) -> jnp.ndarray:
    """Convert boolean mask to integer indices."""
    return jnp.nonzero(mask, size=mask.shape[0])[0]


@jax.jit
def get_active_states(state: State, indices: jnp.ndarray) -> State:
    """Get states for active agents only."""
    return jax.tree.map(lambda x: x[indices], state)


@jax.jit
def place_actions(
    x_actions: jnp.ndarray, 
    y_actions: jnp.ndarray, 
    agent_x: jnp.ndarray, 
    agent_y: jnp.ndarray,
    indices: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Place agent actions in the correct positions."""
    x_actions = x_actions.at[indices].set(agent_x)
    y_actions = y_actions.at[indices].set(agent_y)
    return x_actions, y_actions

def compute_rectangle_schedules(
    num_agents: int,
    num_opponents: int,
    rounds_per_matchup: int,
    team: int,
) -> jnp.ndarray:
    matchups = []
    for i in range(num_agents):
        for j in list(range(num_agents, num_agents + num_opponents)):
            matchups.append((i, j))
            matchups.append((j, i))
    
    schedule = jnp.zeros((num_agents + num_opponents, len(matchups)), dtype=bool)
    for ii, (i, j) in enumerate(matchups):
        if team == 1:
            schedule = schedule.at[j, ii].set(True)
        else:
            schedule = schedule.at[i, ii].set(True)
    schedule = jnp.repeat(schedule, rounds_per_matchup, axis=1)
    return schedule

def compute_square_schedules(
    num_agents: int,
    rounds_per_matchup: int,
    team: int,
) -> jnp.ndarray:
    matchups = list(combinations(range(num_agents), 2))    
    schedule = jnp.zeros((num_agents, len(matchups)), dtype=bool)
    for ii, (i, j) in enumerate(matchups):
        if team == 1:
            schedule = schedule.at[j, ii].set(True)
        else:
            schedule = schedule.at[i, ii].set(True)
    schedule = jnp.repeat(schedule, rounds_per_matchup, axis=1)
    return schedule

def batch_act(
    state: State,
    agents: list,
    agent_schedules: jnp.ndarray,
    team: int,
    key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run a batch of agents and combine their actions.
    
    Args:
        state: Current game state
        agents: List of agent modules
        agent_schedules: Boolean array indicating which agents are active
        team: Team identifier (1 or 2)
        key: Random key for stochastic operations
        
    Returns:
        Tuple of x and y actions for all agents
    """
    batch_size, num_agents = state.x1.shape if team == 1 else state.x2.shape
    
    # Initialize action arrays with zeros
    x_actions = jnp.zeros((batch_size, num_agents))
    y_actions = jnp.zeros((batch_size, num_agents))
    
    # Split key for each agent
    keys = jax.random.split(key, len(agents))
    
    for i, (agent, agent_key) in enumerate(zip(agents, keys)):
        # Get the agents actions for the rounds where it's active
        mask = agent_schedules[i]
        indices = get_indices(mask)

        # Get active states and compute actions
        active_states = get_active_states(state, indices)
        agent_x_actions, agent_y_actions = agent.act(
            t=state.t,
            key=agent_key,
            ally_x=active_states.x1 if team == 1 else active_states.x2,
            ally_y=active_states.y1 if team == 1 else active_states.y2,
            ally_vx=active_states.vx1 if team == 1 else active_states.vx2,
            ally_vy=active_states.vy1 if team == 1 else active_states.vy2,
            ally_health=active_states.health1 if team == 1 else active_states.health2,
            enemy_x=active_states.x1 if team == 2 else active_states.x2,
            enemy_y=active_states.y1 if team == 2 else active_states.y2,
            enemy_vx=active_states.vx1 if team == 2 else active_states.vx2,
            enemy_vy=active_states.vy1 if team == 2 else active_states.vy2,
            enemy_health=active_states.health1 if team == 2 else active_states.health2
        )
        # Place actions in correct positions
        x_actions, y_actions = place_actions(
            x_actions, y_actions, 
            agent_x_actions, agent_y_actions, 
            indices,
        )
    
    return x_actions, y_actions
