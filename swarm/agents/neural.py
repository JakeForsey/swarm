from typing import Tuple

import jax
import jax.numpy as jnp

from swarm.env import State

NUM_FEATURES = 5
NUM_HIDDEN = 32
NUM_OUTPUTS = 2
LAYERS = [
    (NUM_FEATURES, NUM_HIDDEN),
    (NUM_HIDDEN, NUM_HIDDEN),
    (NUM_HIDDEN, NUM_HIDDEN),
    (NUM_HIDDEN, NUM_OUTPUTS),
]
NUM_WEIGHTS = sum(
    [
        in_features * out_features + out_features
        for in_features, out_features in LAYERS
    ]
)

WEIGHTS = jax.random.normal(
    jax.random.PRNGKey(0),
    (NUM_WEIGHTS,)
)
# WEIGHTS = jnp.load("results/optimizer/best_0.16875000298023224.npy")

@jax.jit
def act(
    t: jnp.ndarray,
    key: jnp.ndarray,
    ally_x: jnp.ndarray,
    ally_y: jnp.ndarray,
    ally_vx: jnp.ndarray,
    ally_vy: jnp.ndarray,
    ally_health: jnp.ndarray,
    enemy_y: jnp.ndarray,
    enemy_x: jnp.ndarray,
    enemy_vx: jnp.ndarray,
    enemy_vy: jnp.ndarray,
    enemy_health: jnp.ndarray,
    weights: jnp.ndarray = WEIGHTS,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    batch_size, num_allies = ally_x.shape
    
    dx = ally_x[:, :, None] - enemy_x[:, None, :]
    dy = ally_y[:, :, None] - enemy_y[:, None, :]
    
    # Handle wrapping
    dx = jnp.where(dx > 0.5, dx - 1.0, dx)
    dx = jnp.where(dx < -0.5, dx + 1.0, dx)
    dy = jnp.where(dy > 0.5, dy - 1.0, dy)
    dy = jnp.where(dy < -0.5, dy + 1.0, dy)
    
    alive_enemies = enemy_health > 0
    distances = jnp.sqrt(dx**2 + dy**2) + 1e-5
    distances = jnp.where(alive_enemies[:, None, :], distances, jnp.inf)
    
    nearest_enemy_idx = jnp.argmin(distances, axis=2)
    batch_idx = jnp.arange(batch_size)[:, None]
    agent_idx = jnp.arange(num_allies)[None, :]
    nearest_enemy_health = enemy_health[batch_idx, nearest_enemy_idx]
    
    nearest_enemy_dx = dx[batch_idx, agent_idx, nearest_enemy_idx]
    nearest_enemy_dy = dy[batch_idx, agent_idx, nearest_enemy_idx]

    X = jnp.concatenate([
        nearest_enemy_dx,
        nearest_enemy_dy,
        ally_health - nearest_enemy_health,
        ally_health,
        nearest_enemy_health,
    ], axis=1)
    X = X.reshape(batch_size * num_allies, NUM_FEATURES)

    offset = 0
    for i, (in_features, out_features) in enumerate(LAYERS):
        W_start = offset
        W_end = offset + in_features * out_features
        W = weights[W_start:W_end].reshape(in_features, out_features)
        b_start = W_end
        b_end = b_start + out_features
        b = weights[b_start:b_end]
        X = jnp.matmul(X, W) + b
        if i < len(LAYERS) - 1:
            # X = jax.nn.relu(X)
            # X = jax.nn.leaky_relu(X)
            X = jax.nn.tanh(X)

        offset = b_end

    y = X.reshape(batch_size, num_allies, 2)
    dx_out = y[:, :, 0]
    dy_out = y[:, :, 1]
    # dx_out = nearest_enemy_dx * y[:, :, 0]
    # dy_out = nearest_enemy_dy * y[:, :, 1]

    return dx_out, dy_out


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from swarm.agents import get_agent
    from swarm.tournament import _run

    class Agent:
        def __init__(self, name: str, weights: jnp.ndarray):
            self.__name__ = name
            self.weights = weights

        def act(
            self,
            state: State,
            team: int,
            key: jax.random.PRNGKey,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            return act(state, team, key, self.weights)

    num_rounds_per_matchup = 32
    pop_size = 64

    from swarm.agents import load_agents, get_agent
    baseline_agents = [
        get_agent("random"),
        get_agent("vortex_swarm_v2"),
        get_agent("predator_swarm"),
        get_agent("hunter_swarm"),
        get_agent("simple"),
    ]
    # baseline_agents = load_agents()
    print(len(baseline_agents))

    from evojax.algo.simple_ga import SimpleGA
    from evojax.algo.open_es import OpenES
    from evojax.algo.cma_jax import CMA_ES_JAX
    from evojax.algo.pgpe import PGPE
    from evojax.algo.ars import ARS

    # optimizer = ARS(
    #     param_size=WEIGHTS.shape[0],
    #     pop_size=pop_size,
    #     init_stdev=0.1,
    # )
    # optimizer = SimpleGA(
    #     param_size=WEIGHTS.shape[0],
    #     pop_size=pop_size,
    #     sigma=0.05,
    # )
    optimizer = OpenES(
        param_size=WEIGHTS.shape[0],
        pop_size=pop_size,
        init_stdev=0.1,
    )
    # optimizer = PGPE(
    #     param_size=WEIGHTS.shape[0],
    #     pop_size=pop_size,
    #     optimizer="clipup",
    #     solution_ranking=False,
    # )
    maxes = []
    means = []
    for step in range(64):
        population_weights = optimizer.ask()
        print(population_weights.shape)
        population = [
            Agent(f"neural_{i}", population_weights[i])
            for i in range(pop_size)
        ]
        results =_run(
            population + baseline_agents,
            num_rounds_per_matchup=num_rounds_per_matchup,
            episode_length=32,
        )

        fitnesses = []
        for individual, weights, result in zip(population, population_weights, results):
            fitnesses.append(result['reward'])
        fitnesses = jnp.array(fitnesses)
        
        best = fitnesses.max().item()
        mean = fitnesses.mean().item()
        maxes.append(best)
        means.append(mean)
        optimizer.tell(fitnesses)
        print(mean, best, step, optimizer.best_params)
        if best == max(maxes):
            jnp.save(f"results/optimizer/best_{best}.npy", optimizer.best_params)

        plt.clf()
        plt.cla()
        plt.plot(maxes, label="max")
        plt.plot(means, label="mean")
        plt.legend()
        plt.savefig(f"results/optimizer/neural/{step}.png")
