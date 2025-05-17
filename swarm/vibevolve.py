"""
Automatically write swarm agents using LLMs and evolution (MAP Elites).

Inspired by:

Jean-Baptiste Mouret & Jeff Clune
Illuminating search spaces by mapping elites
arxiv. 2015
https://arxiv.org/abs/1504.04909

Novikov et al.
AlphaEvolve: A coding agent for scientific and algorithmic discovery.
Technical report. 2025.
https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf

Romera-Paredes, B., Barekatain, M., Novikov, A. et al. 
Mathematical discoveries from program search with large language models.
Nature 625, 468–475 (2024).
https://doi.org/10.1038/s41586-023-06924-6
"""
from functools import partial
import json
from multiprocessing import get_context
import os
import uuid
import random
import time
import types
from typing import NamedTuple

import requests

from swarm.agents import load_agents, get_agent
from swarm import tournament

# OPPONENTS = load_agents()
OPPONENTS = [
    get_agent("simple"),
    get_agent("vortex_swarm"),
    get_agent("hunter_swarm"),
    get_agent("predator_boid"),
    get_agent("boid"),
    get_agent("chaser"),
    get_agent("concave_swarm"),
    get_agent("center"),
    get_agent("pairs"),
    get_agent("random"),
]

TMP_AGENT_NAME = "tmp_agent"

# ---------------------------------- Workers ----------------------------------

PROMPT = f"""\
Please develop a competitive agent in python (using jax) that competes in a two player environment.

The Rules:
 * Each player gets 32 pieces they control
 * If pieces from opposing teams collide they both take damage
 * Pieces regenerate some health each step
 * Each agent returns a dvx and a dvy; deltas that the environment will apply to their peices velocities (vx and vy)

Here is an **example** implementation of the `act` interface:
```python
import jax
import jax.numpy as jnp

@jax.jit
def act(
    # Time step jnp.ndarray[batch_size] 
    t,
    # Jax random key
    key, 
    # Positions, velocities and health jnp.ndarray[batch_size, num_agents]
    ally_x,
    ally_y,
    ally_vx,
    ally_vy,
    ally_health,
    enemy_y,
    enemy_x,
    enemy_vx,
    enemy_vy,
    enemy_health,
):
    batch_size, num_agents = ally_x.shape
    # TODO: Implement a strategy here instead of zeros
    dvx = jnp.zeros((batch_size, num_agents))
    dvy = jnp.zeros((batch_size, num_agents))
    return dvx, dvy
```

Remember:
 * All jax code should be vectorized so avoid if statements and for loops
 * All the inputs have a batch axis (B)
 * All the pieces inputs have a pieces axis (32)
 * Your full implementation should be contained in a single python code block"""

_worker_host = None

def init_worker(host):
    global _worker_host
    _worker_host = host
    print(f"Initialized worker with host: {_worker_host} (pid={os.getpid()})")

# ------------------------------ Llama Requests -------------------------------

def request_model(host):
    response = requests.get(
        f"http://{host}/models"
    )
    data = response.json()["data"]
    model = data[0]["id"].split("/")[-1]
    return model

def request_completion(host, prompt):
    response = requests.post(
        f"http://{host}/v1/chat/completions",
        json={
            "messages":[
                {"role": "user","content": prompt},
            ],
            "stream": False,
            "cache_prompt": False,
            "samplers": "edkypmxt",
            "temperature": 0.8,
            "dynatemp_range": 0,
            "dynatemp_exponent": 1, 
            "top_k": 40,
            "top_p": 0.95,
            "min_p": 0.05,
            "typical_p": 1,
            "xtc_probability": 0,
            "xtc_threshold": 0.1,
            "repeat_last_n": 64,
            "repeat_penalty": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "dry_multiplier": 0,
            "dry_base": 1.75,
            "dry_allowed_length": 2,
            "dry_penalty_last_n": -1,
            "max_tokens": -1,
            "timings_per_token": False,
        },
    )
    data = response.json()
    choices = data["choices"]
    choice = choices[0]
    completion = choice["message"]["content"]
    return completion

# --------------------------------- MAP Elites --------------------------------

class Elite(NamedTuple):
    completion_id: str
    reward: float
    src: str

def get_elites(run_id):
    history_path = "results/vibevolve/rewards.jsonl"
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            lines = f.readlines()
    else:
        lines = []

    records = [json.loads(line) for line in lines]
    records = [
        record for record in records
        if record["run_id"] == run_id
    ]
    
    elites_by_niche = {}
    for record in records:
        completion_id = record["completion_id"]
        src = src_from_history(record["run_id"], completion_id)
        
        for niche in record["results"]:
            niche_name = niche["name"]

            reward = niche["reward"]
            reward = reward if niche_name == TMP_AGENT_NAME else -reward
            if reward == -1.0:
                continue
            
            elite = elites_by_niche.get(niche_name, Elite(None, -1.0, None))
            if reward >= elite.reward:
                elite = Elite(completion_id, reward, src)
                elites_by_niche[niche_name] = elite
    
    for niche_name, elite in elites_by_niche.items():
        print(f"{niche_name:>20}: {elite.reward:.5f} [{elite.completion_id}]")
    
    return elites_by_niche

# ---------------------------------- History ----------------------------------

def completion_directory(run_id, completion_id):
    return f"results/vibevolve/{run_id}/{completion_id}"

def persist_in_history(
        run_id,
        completion_id,
        parent_completion_id,
        start_timestamp,
        results,
        completion,
):
    tmp_agent_result = [
        result for result in results
        if result["name"] == TMP_AGENT_NAME
    ]
    reward = tmp_agent_result[0]["reward"]

    directory = completion_directory(run_id, completion_id)
    os.makedirs(directory, exist_ok=True)
    
    with open(f"{directory}/completion.txt", "w") as f:
        f.write(completion)
    
    with open(f"results/vibevolve/rewards.jsonl", "a") as f:
        data = {
            "run_id": run_id,
            "completion_id": completion_id,
            "timestamp": start_timestamp,
            "parent_completion_id": parent_completion_id,
            "reward": reward,
            "results": results,
            "host": _worker_host,
        }
        f.write(json.dumps(data) + "\n")

def src_from_history(run_id, completion_id):
    directory = completion_directory(run_id, completion_id)
    path = f"{directory}/completion.txt"
    with open(path, "r") as f:
        completion = f.read()
    return src_from_completion(completion)

def src_from_completion(completion):
    if "```python" not in completion:
        return None
    _, src, *__ = completion.split("```python")
    if "```" not in src:
        return None
    src, *_ = src.split("```")
    return src

def agent_from_completion(completion):
    src = src_from_completion(completion)
    if src is None:
        return None
    try:
        agent = types.ModuleType(TMP_AGENT_NAME)
        exec(src, agent.__dict__)
    except:
        return None
    return agent

# ---------------------------------- Reward -----------------------------------

def run_tournament(
        completion,
        opponents,
        num_rounds_per_matchup,
        episode_length,
    ):
    try:
        agent = agent_from_completion(completion)
        results = tournament.run(
            [agent],
            opponents,
            num_rounds_per_matchup=num_rounds_per_matchup,
            episode_length=episode_length,
        )
        for result in results:
            # Convert single element jax array into float
            result["reward"] = float(result["reward"])
    except Exception:
        # If the agent failed, it gets very lowest possible rewards
        results = [
            {"name": opp.__name__.split(".")[-1], "reward": 1.0}
            for opp in opponents
        ]
        results.append({"name": TMP_AGENT_NAME, "reward": -1.0})

    return results

# --------------------------------- VibEvolve ---------------------------------

def build_prompt(run_id):
    elites = get_elites(run_id)
    if elites:
        elite = random.choice(list(elites.values()))
        parent_completion_id = elite.completion_id
        sampled_context = f"""\
Attempt to improve the following agent:
```python
{elite.src}
```
"""
    else:
        parent_completion_id = None
        sampled_context = ""
    
    prompt = "\n".join([PROMPT, sampled_context, "/no_think"])    
    return prompt, parent_completion_id

def one_vibe(run_id, num_rounds_per_matchup, episode_length, index):
    global _worker_host
    worker_prefix = f"[{os.getpid()}] [{_worker_host}]"

    start_timestamp = time.time()
    completion_id = str(start_timestamp)

    print(f"{worker_prefix} Requesting model...")
    model = request_model(_worker_host)
    print(f"[{worker_prefix}] {model=}")

    print(f"[{worker_prefix}] Building prompt...")
    prompt, parent_completion_id = build_prompt(run_id)

    print(f"[{worker_prefix}] [{_worker_host}] Requesting completion...")
    completion = request_completion(_worker_host, prompt)

    print(f"[{worker_prefix}] Running tournament...")
    results = run_tournament(
        completion=completion,
        opponents=OPPONENTS,
        num_rounds_per_matchup=num_rounds_per_matchup,
        episode_length=episode_length,
    )

    print(f"[{os.getpid()}] Persisting completion in history...")
    persist_in_history(
        run_id=run_id,
        completion_id=completion_id,
        parent_completion_id=parent_completion_id,
        start_timestamp=start_timestamp,
        results=results,
        completion=completion,
    )

def run(hosts, num_rounds_per_matchup, episode_length):
    run_id = str(uuid.uuid1()).split("-")[0]
    print(f"{run_id=}")

    ctx = get_context("spawn")
    with ctx.Pool(processes=len(hosts)) as pool:
        for host in hosts:
            pool.apply_async(init_worker, args=(host,))
        func = partial(one_vibe, run_id, num_rounds_per_matchup, episode_length)
        pool.map(func, range(256))
