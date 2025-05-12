import time
from typing import Callable
import jax
import jax.numpy as jnp

from swarm.agents import load_agents
from swarm.env import SwarmEnv
from swarm.batch import batch_act, compute_square_schedules, compute_rectangle_schedules

def run(
        agents: list[Callable] = None,
        opponents: list[Callable] = None,
        num_rounds_per_matchup: int = 256,
        episode_length: int = 128
    ):
    if agents is None:
        assert opponents is None
        print("[init] Loading agents...")
        agents = load_agents()
        print("[init] Computing agent schedules...")
        agent_schedules1 = compute_square_schedules(len(agents), num_rounds_per_matchup, 1)
        agent_schedules2 = compute_square_schedules(len(agents), num_rounds_per_matchup, 2)
    else:
        assert opponents is not None
        print("[init] Computing agent schedules...")
        agent_schedules1 = compute_rectangle_schedules(len(agents), len(opponents), num_rounds_per_matchup, 1)
        agent_schedules2 = compute_rectangle_schedules(len(agents), len(opponents), num_rounds_per_matchup, 2)
        agents = agents + opponents
    return _run(agents, agent_schedules1, agent_schedules2, episode_length)

def _run(
        agents: list[Callable],
        agent_schedules1: jnp.array, 
        agent_schedules2: jnp.array,
        episode_length: int = 128,
    ):
    batch_size = agent_schedules1.shape[1]

    print("[init] Initializing environment...")
    env = SwarmEnv(batch_size=batch_size, episode_length=episode_length)

    print("[init] Jitting environment, agents and running warmup step...")
    state = env.reset()
    x_action1, y_action1 = batch_act(state, agents, agent_schedules1, 1, jax.random.PRNGKey(0))
    x_action2, y_action2 = batch_act(state, agents, agent_schedules2, 2, jax.random.PRNGKey(0))
    state, reward = env.step(state, x_action1, y_action1, x_action2, y_action2)
    state = env.reset()

    print("[tournament] Starting tournament...")
    start = time.perf_counter()
    keys1 = jax.random.split(jax.random.PRNGKey(0), env.episode_length)
    keys2 = jax.random.split(jax.random.PRNGKey(1), env.episode_length)
    for step, key1, key2 in zip(range(1, env.episode_length + 1), keys1, keys2):
        print(f"[tournament] Step {step}/{env.episode_length}")
        x_action1, y_action1 = batch_act(state, agents, agent_schedules1, 1, key1)
        x_action2, y_action2 = batch_act(state, agents, agent_schedules2, 2, key2)
        state, reward = env.step(state, x_action1, y_action1, x_action2, y_action2)
        
        if step == env.episode_length:
            # Create a dictionary of agent name -> reward
            reward_summary = []
            for i, agent in enumerate(agents):
                agent_name = agent.__name__.split(".")[-1]
                team1_reward = reward[agent_schedules1[i]]
                team2_reward = -1 * reward[agent_schedules2[i]]
                rewards = jnp.concatenate([team1_reward, team2_reward])
                reward_summary.append(
                    {
                        "name": agent_name,
                        "reward": rewards.mean(),
                    }
                )

            for summary in sorted(reward_summary, key=lambda x: x["reward"], reverse=True):
                print(f"{summary['name']:>20} reward: {summary['reward']:.2f}")
    
    end = time.perf_counter()
    tournament_duration = end - start
    
    print(f"[tournament] Tournament complete...")
    print(f"[tournament] Time taken: {tournament_duration:.2f} seconds")
    print(f"[tournament] Steps per second: {(env.episode_length * env.batch_size) / tournament_duration:,.0f}")

    return reward_summary
