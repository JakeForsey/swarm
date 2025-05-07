import time

import jax
import jax.numpy as jnp

from swarm.agents import load_agents
from swarm.env import SwarmEnv
from swarm.batch import batch_act, compute_agent_schedules


def main():
    num_rounds_per_matchup = 512
    episode_length = 128
    agents = load_agents()
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
