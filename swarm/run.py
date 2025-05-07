import importlib
from itertools import combinations
import pkgutil
import time

import jax
import jax.numpy as jnp

from swarm import agents as agents_module
from swarm.env import SwarmEnv, State


def compute_agent_schedules(num_agents: int, rounds_per_matchup: int, team: int) -> jnp.ndarray:
    matchups = list(combinations(range(num_agents), 2))
    schedule = jnp.zeros((num_agents, len(matchups)), dtype=bool)
    for ii, (i, j) in enumerate(matchups):
        if team == 1:
            schedule = schedule.at[j, ii].set(True)
        else:
            schedule = schedule.at[i, ii].set(True)
    schedule = jnp.repeat(schedule, rounds_per_matchup, axis=1)
    return schedule


@jax.jit
def get_indices(mask: jnp.ndarray) -> jnp.ndarray:
    """Convert boolean mask to integer indices."""
    return jnp.nonzero(mask, size=mask.shape[0])[0]


@jax.jit
def get_active_states(state: State, indices: jnp.ndarray) -> State:
    """Get states for active agents only."""
    return jax.tree.map(lambda x: x[indices], state)


@jax.jit
def place_actions(x_actions: jnp.ndarray, y_actions: jnp.ndarray, 
                 agent_x: jnp.ndarray, agent_y: jnp.ndarray,
                 indices: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Place agent actions in the correct positions."""
    x_actions = x_actions.at[indices].set(agent_x)
    y_actions = y_actions.at[indices].set(agent_y)
    return x_actions, y_actions


def batch_act(state: State, agents: list, agent_schedules: jnp.ndarray, team: int, key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
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


def main():
    num_rounds_per_matchup = 1024
    episode_length = 128

    agents = [
        importlib.import_module(f"swarm.agents.{info.name}")
        for info in pkgutil.iter_modules(agents_module.__path__)
    ]
    num_agents = len(agents)
    agent_schedules1 = compute_agent_schedules(num_agents, num_rounds_per_matchup, 1)
    agent_schedules2 = compute_agent_schedules(num_agents, num_rounds_per_matchup, 2)
    batch_size = agent_schedules1.shape[1]
    env = SwarmEnv(batch_size=batch_size, episode_length=episode_length)
    state = env.reset()

    print(f"Batch size: {batch_size}")
    print(f"Num agents: {num_agents}")
    print(f"Num rounds per matchup: {num_rounds_per_matchup}")
    print(f"Episode length: {episode_length}")

    start = time.perf_counter()
    keys1 = jax.random.split(jax.random.PRNGKey(0), env.episode_length)
    keys2 = jax.random.split(jax.random.PRNGKey(1), env.episode_length)
    for step, key1, key2 in zip(range(env.episode_length), keys1, keys2):
        print(f"Step {step}/{env.episode_length}")
        x_action1, y_action1 = batch_act(state, agents, agent_schedules1, 1, key1)
        x_action2, y_action2 = batch_act(state, agents, agent_schedules2, 2, key2)
        state, reward = env.step(state, x_action1, y_action1, x_action2, y_action2)
        
        if step == env.episode_length - 1:
            # Create a dictionary of agent name -> reward
            reward_summary = {}
            for i, agent in enumerate(agents):
                team1_reward = reward[agent_schedules1[i]]
                team2_reward = -1 * reward[agent_schedules2[i]]
                rewards = jnp.concatenate([team1_reward, team2_reward])
                reward_summary[agent.__name__] = rewards.mean()

            for agent, reward in sorted(reward_summary.items(), key=lambda x: x[1], reverse=True):
                print(f"{agent:>30} reward: {reward:.2f}")
    
    end = time.perf_counter()
    print(f"Steps per second: {(env.episode_length * env.batch_size) / (end - start):,.0f}")


if __name__ == "__main__":
    main()
