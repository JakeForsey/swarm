import jax
import jax.numpy as jnp
from itertools import combinations
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


def compute_agent_schedules(num_agents: int, rounds_per_matchup: int, team: int) -> jnp.ndarray:
    """Compute the schedule of which agents are active in each round.
    
    Args:
        num_agents: Total number of agents
        rounds_per_matchup: Number of rounds for each agent matchup
        team: Team identifier (1 or 2)
        
    Returns:
        Boolean array of shape (num_agents, total_rounds) indicating which agents
        are active in each round
    """
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
        agent_x_actions, agent_y_actions = agent.act(active_states, team, agent_key)
        
        # Place actions in correct positions
        x_actions, y_actions = place_actions(
            x_actions, y_actions, 
            agent_x_actions, agent_y_actions, 
            indices,
        )
    
    return x_actions, y_actions
