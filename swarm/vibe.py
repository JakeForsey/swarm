from functools import partial
from multiprocessing import get_context
import os
import uuid

import requests

from swarm.rl import PROMPT, reward_tournament

HOSTS = ["cortex1:8080", "cortex2:8080"]

REQUEST_BODY = {
    "messages":[
        {"role": "user","content": PROMPT}
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
    "timings_per_token":False
}

_worker_host = None

def init_worker(host):
    global _worker_host
    _worker_host = host
    print(f"Initialized worker with host: {_worker_host} (pid={os.getpid()})")

def one_vibe(run_id, index):
    global _worker_host
    step = 0
    model = "unsloth/Qwen3-14B-GGUF"
    llama_server_cmd = f"llama-server -hf {model} --ctx-size 10000"

    response = requests.post(
        f"http://{_worker_host}/v1/chat/completions",
        json=REQUEST_BODY,
    )
    data = response.json()

    choices = data["choices"]
    choice = choices[0]
    completion = choice["content"]

    reward_tournament(
        completion,
        run_id,
        step,
        index,
        metadata={
            "algorithm": "topk",
            "model": model,
            "llama_server_command": llama_server_cmd,
        },
    )

def run():
    run_id = str(uuid.uuid1()).split("-")[0]
    print(f"{run_id=}")
    ctx = get_context("spawn")
    with ctx.Pool(
        processes=len(HOSTS),
        initializer=init_worker,
        initargs=(HOSTS[0],),  # This will be overridden below
    ) as pool:
        for host in HOSTS:
            pool.apply_async(init_worker, args=(host,))

        func = partial(one_vibe, run_id)
        pool.map(func, range(256))
