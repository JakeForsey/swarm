import json
import time
import random
import uuid

from redis import Redis
from rq import Queue, Retry

from swarm.selfplay.worker import mutate_and_evaluate


def run(
    num_rounds_per_matchup: int,
    episode_length: int,
    pop_size: int,
    generations: int,
):
    run_id = str(uuid.uuid1()).split("-")[0]

    q = Queue(connection=Redis())
    population = []
    for generation in range(generations):
        print(generation)
        jobs = []
        population_srcs = [x["src"] for x in population]
        for _ in range(pop_size):
            parent_srcs = random.sample(population_srcs, k=min(len(population_srcs), 2))
            job = q.enqueue(
                mutate_and_evaluate,
                args=(
                    parent_srcs,
                    population_srcs,
                    num_rounds_per_matchup,
                    episode_length,
                ),
                job_timeout=60 * 5,
                retry=Retry(max=3),
            )
            jobs.append(job)

        time.sleep(1)

        results = []
        while len(jobs) > 0:
            job = jobs.pop(0)
            if job.is_queued or job.is_started or job.is_deferred:
                jobs.append(job)
            else:
                if job.result is not None:
                    results.append(job.result)
                else:
                    print(job)
            time.sleep(0.1)

        base_directory = f"results/evolve/{run_id}"
        directory = f"{base_directory}/{generation}/"
        import os

        os.makedirs(directory, exist_ok=True)
        for result in results:
            with open(os.path.join(directory, result["id"] + ".json"), "w") as f:
                json.dump({k: v for k, v in result.items() if k != "src"}, f)
            with open(os.path.join(directory, result["id"] + ".py"), "w") as f:
                f.write(result["src"])

        last_population = population.copy()
        last_population.sort(key=lambda x: sum(x["scores"].values()), reverse=True)

        next_population = [result for result in results if result["success"]]
        next_population.sort(key=lambda x: sum(x["scores"].values()), reverse=True)

        num_survivors = min(len(last_population), pop_size // 2)
        num_offspring = pop_size - num_survivors
        population = last_population[:num_survivors] + next_population[:num_offspring]
