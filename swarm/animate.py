import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from swarm.agents import boid
from swarm.agents import random
from swarm.agents import chaser
from swarm.agents import fleeing
from swarm.agents import smart_boid
from swarm.agents import clusters
from swarm.agents import predator_boid
from swarm.agents import ring_swarm
from swarm.agents import vortex_swarm
from swarm.agents import squad_swarm
from swarm.agents import center_swarm
from swarm.agents import static_swarm
from swarm.agents import train_swarm
from swarm.env import SwarmEnv


class SwarmSimulator:
    def __init__(self):
        self.key = jax.random.PRNGKey(0)
        self.env = SwarmEnv(batch_size=2)

        plt.switch_backend('Agg')
        self.fig, self.ax = plt.subplots(figsize=(8, 8), dpi=80)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.scatter1 = self.ax.scatter([], [], c='red', s=50, alpha=0.6, edgecolors='none')
        self.scatter2 = self.ax.scatter([], [], c='blue', s=50, alpha=0.6, edgecolors='none')
    
    def update(self, frame):
        """Update function for animation."""
        team1_key, team2_key, self.key = jax.random.split(self.key, 3)
        agent1 = train_swarm
        agent2 = vortex_swarm
        x_action1, y_action1 = agent1.act(self.state, team=1, key=team1_key)
        x_action2, y_action2 = agent2.act(self.state, team=2, key=team2_key)

        self.state, _ = self.env.step(self.state, x_action1, y_action1, x_action2, y_action2)
        
        alive1 = self.state.health1 > 0
        alive2 = self.state.health2 > 0

        positions1 = jnp.stack([self.state.x1, self.state.y1], axis=-1)
        positions2 = jnp.stack([self.state.x2, self.state.y2], axis=-1)

        self.scatter1.set_offsets(positions1[0][alive1[0]])
        self.scatter2.set_offsets(positions2[0][alive2[0]])

        self.ax.set_title(f"{agent1.__name__}={self.state.health1[0].sum():.1f} vs {agent2.__name__}={self.state.health2[0].sum():.1f}")

        return [self.scatter1, self.scatter2]

    def run(self, output_file='swarm_simulation.gif'):
        self.state = self.env.reset()
        anim = FuncAnimation(
            self.fig, 
            self.update, 
            frames=self.env.episode_length,
            blit=True,
            repeat=False
        )
        anim.save(output_file, writer=PillowWriter(fps=20, bitrate=1800))
        print(f"Animation saved to {output_file}")


if __name__ == "__main__":
    simulator = SwarmSimulator()
    simulator.run()
