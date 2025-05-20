import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from swarm.agents import get_agent
from swarm.env import SwarmEnv
from swarm.batch import compute_rectangle_schedules, batch_act


def create_animation(states, agent1_name, agent2_name, filename):
    """Create and save an animation of the battle."""
    agent1_colour = "blue"
    agent2_colour = "red"

    # Create figure with two subplots
    fig = plt.figure(figsize=(8, 8.5))  # Reduce overall figure size
    gs = fig.add_gridspec(2, 1, height_ratios=[8, 0.5])

    # Plot agent positions
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_aspect("equal")
    ax1.grid(False)
    agent1_scatter = ax1.scatter([], [], c=agent1_colour, alpha=0.5)
    agent2_scatter = ax1.scatter([], [], c=agent2_colour, alpha=0.5)

    # Add title with colored team names
    ax1.text(
        0.05,
        1.01,
        agent1_name,
        color=agent1_colour,
        ha="left",
        va="bottom",
        fontsize=18,
        transform=ax1.transAxes,
    )
    ax1.text(
        0.95,
        1.01,
        agent2_name,
        color=agent2_colour,
        ha="right",
        va="bottom",
        fontsize=18,
        transform=ax1.transAxes,
    )

    # Health bar plot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_xlim(0.0, 1.0)
    ax2.axis("off")
    ax2.barh(
        [0.65, 0.35],
        [1, 1],
        color="lightgray",
        height=0.25,
        alpha=0.3,
    )
    agent1_health, agent2_health = ax2.barh(
        [0.65, 0.35],
        [0, 0],
        color=[agent1_colour, agent2_colour],
        height=0.25,
        alpha=0.8,
    )
    fig.subplots_adjust(
        top=0.94, bottom=0.01, left=0.01, right=0.99, hspace=0.01, wspace=0.01
    )

    def update(frame):
        # Update team positions
        agent1_alive = states[frame].health1[0] > 0
        agent2_alive = states[frame].health2[0] > 0

        # Get positions of alive agents only
        agent1_positions = jnp.column_stack(
            [states[frame].x1[0][agent1_alive], states[frame].y1[0][agent1_alive]]
        )
        agent2_positions = jnp.column_stack(
            [states[frame].x2[0][agent2_alive], states[frame].y2[0][agent2_alive]]
        )
        agent1_scatter.set_offsets(agent1_positions)
        agent2_scatter.set_offsets(agent2_positions)

        # Calculate health percentages
        total_health1 = jnp.sum(states[frame].health1[0]) / len(
            states[frame].health1[0]
        )
        total_health2 = jnp.sum(states[frame].health2[0]) / len(
            states[frame].health2[0]
        )

        # Update health bars
        agent1_health.set_width(total_health1)
        agent2_health.set_width(total_health2)

        return [agent1_scatter, agent2_scatter, agent1_health, agent2_health]

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(states), blit=True)

    # Save animation
    anim.save(f"results/animations/{filename}.gif", writer="pillow", fps=20)
    plt.close()


def run_match(env: SwarmEnv, agent1, agent2, num_agents: int = 32):
    """Run a single match between two agents and return the states."""
    # Compute agent schedules
    agent_schedules1 = compute_rectangle_schedules(1, 1, 1, 1)
    agent_schedules2 = compute_rectangle_schedules(1, 1, 1, 2)

    # Reset environment
    state = env.reset()
    states = [state]

    # Generate keys for the episode
    keys1 = jax.random.split(jax.random.PRNGKey(0), env.episode_length)
    keys2 = jax.random.split(jax.random.PRNGKey(1), env.episode_length)

    # Run the episode
    for key1, key2 in zip(keys1, keys2):
        # Get actions from both agents
        x_action1, y_action1 = batch_act(state, [agent1], agent_schedules1, 1, key1)
        x_action2, y_action2 = batch_act(state, [agent2], agent_schedules2, 2, key2)

        # Step the environment
        state, reward = env.step(state, x_action1, y_action1, x_action2, y_action2)
        states.append(state)

    return states, reward


def run(agent1_name: str, agent2_name: str, episode_length: int = 128):
    os.makedirs("results/animations", exist_ok=True)

    agent1 = get_agent(agent1_name)
    agent2 = get_agent(agent2_name)

    env = SwarmEnv(batch_size=1, episode_length=episode_length)
    states, _ = run_match(env, agent1, agent2)

    filename = f"{agent1_name}_vs_{agent2_name}"
    create_animation(states, agent1_name, agent2_name, filename)
