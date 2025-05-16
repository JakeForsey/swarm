from functools import partial
import hashlib
import json
from multiprocessing import get_context
import os
import uuid
import random

import requests

from swarm.rl import PROMPT, reward_tournament, _parse_src

# ---------------------------------- Workers ----------------------------------

HOSTS = ["cortex1:8080", "cortex2:8080"]

_worker_host = None

def init_worker(host):
    global _worker_host
    _worker_host = host
    print(f"Initialized worker with host: {_worker_host} (pid={os.getpid()})")

# ------------------------------ Llama Requests -------------------------------

def _completion_request_body(prompt):
    return {
        "messages":[
            {"role": "user","content": prompt}
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
    }

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
        json=_completion_request_body(prompt),
    )
    data = response.json()
    choices = data["choices"]
    choice = choices[0]
    completion = choice["message"]["content"]
    return completion

# --------------------------------- MAP Elites --------------------------------

def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()

def compute_niches(run_id):
    with open("results/logdir/rewards.jsonl", "r") as f:
        lines = f.readlines()
    records = [json.loads(line) for line in lines]
    records = [
        record for record in records
        if record["run_id"] == run_id
    ]
    # Crude map elites algo over reward space
    # opp: (reward, ...)
    niches = {}
    for record in records:
        for result in record["results"]:
            reward = result["reward"] if result["name"] == "tmp_agent" else -result["reward"]
            if reward == -1.0:
                continue
            elite_reward, _ = niches.get(result["name"], (-1.0, None))
            if reward >= elite_reward:
                src = read_src(record["run_id"], record["step"], record["index"])
                niches[result["name"]] = (reward, src)
    for name, (reward, src) in niches.items():
        print(f"{name:>20}: {reward} [{hash_text(src)}]")
    return niches

def sample_src(niches):
    srcs = [src for (_, src) in niches.values()]
    return random.choice(srcs)

def build_prompt(run_id, thinking):
    niches = compute_niches(run_id)
    if niches:
        src = sample_src(niches)
        sampled_context = f"""\
Attempt to improve the following agent:
```python
{src}
```
"""
    else:
        sampled_context = ""
    thinking = "/think" if thinking else "/no_think"
    return "\n".join([PROMPT, sampled_context, thinking])

# ------------------------------- Chunk Source --------------------------------

def read_src(run_id, step, index):
    path = f"results/logdir/{run_id}/completions/{step}/{index}/completion.txt"
    with open(path, "r") as f:
        completion = f.read()
    return _parse_src(completion)

def one_vibe(run_id, index):
    global _worker_host
    step = 0
    thinking = False

    print(f"[{os.getpid()}] Requesting model...")
    model = request_model(_worker_host)
    print(f"[{os.getpid()}] {model=}")

    print(f"[{os.getpid()}] Building prompt...")
    prompt = build_prompt(run_id, thinking)

    print(f"[{os.getpid()}] Requesting completion {_worker_host}...")
    completion = request_completion(
        _worker_host,
        prompt,
    )
    print(f"[{os.getpid()}] Running tournament...")

    # TODO: Timeout the tournament if it takes too long
    reward_tournament(
        completion,
        run_id,
        step,
        index,
        metadata={
            "algorithm": "vibevolve",
            "model": model,
            "thinking": thinking,
            "host": _worker_host,
        },
    )


def run():
    # run_id = str(uuid.uuid1()).split("-")[0]
    # start = 0

    run_id = "45ff689d"
    start = 64
    print(f"{run_id=}")

    ctx = get_context("spawn")
    with ctx.Pool(
        processes=len(HOSTS),
        initializer=init_worker,
        # This will be overridden below
        initargs=(HOSTS[0],),
    ) as pool:
        for host in HOSTS:
            pool.apply_async(init_worker, args=(host,))
        func = partial(one_vibe, run_id)
        pool.map(func, range(start, start + 256))
