import time

import jax

from swarm import tournament
from swarm.agents import get_agent


def run():
    device = jax.default_backend()

    episode_length = 128
    agents = [get_agent("random")]
    opponents = [get_agent("random")]

    for i in range(20):
        num_rounds_per_matchup = 2**i
        # Warmup run to pre-jit
        tournament.run(
            agents,
            opponents,
            num_rounds_per_matchup=1,
            episode_length=1,
        )

        start = time.perf_counter()
        tournament.run(
            agents,
            opponents,
            num_rounds_per_matchup=num_rounds_per_matchup,
            episode_length=episode_length,
        )
        seconds = time.perf_counter() - start

        batch_size = len(agents) * len(opponents) * 2 * num_rounds_per_matchup
        steps = batch_size * episode_length
        steps_per_second = round(steps / seconds)
        print(
            f"{device=} | {batch_size=:>10,} | {steps=:>12,} | {seconds=:>6.3f} | {steps_per_second=:>11,}"
        )
