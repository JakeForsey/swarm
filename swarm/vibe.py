import json
import os
import uuid
import time
import traceback

import requests

from swarm import tournament

def write_src(src, agent):
    file_path = f"swarm/agents/{agent}.py"
    with open(file_path, "w") as f:
        f.write(src)

def read_src(file_path):
    with open(file_path, "r") as f:
        return f.read()

def read_agent(agent):
    return read_src(f"swarm/agents/{agent}.py")

SYSTEM_PROMPT = f"""\
You are a helpful assistant.
"""

VIBE_SYSTEM_PROPMPT = f"""\
/no_think We are going to write a competitive agent that controls boids to swarm against an opponents boids.

Example agent 1:
```python
{read_agent('random')}
```

Example agent 2:
```python
{read_agent('simple')}
```

Example agent 3:
```python
{read_agent('vortex_swarm_v2')}
```

Example agent 4:
```python
{read_agent('boid')}
```

Return your whole agent as **concise** & **compact** standalone module by implementing the following function:
```python
import jax
import jax.numpy as jnp

@jax.jit
def act(
    t,
    key,
    # shapes: [batch, n_boids]
    ally_x,
    ally_y,
    ally_vx,
    ally_vy,
    ally_health,
    enemy_y,
    enemy_x,
    enemy_vx,
    enemy_vx,
    enemy_health,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    ...
```

 * Always use `jnp` instead of `np`
 * JAX arrays are immutable and do not support in-place item assignment. Instead of x[idx] = y, use x = x.at[idx].set(y)
 * Do not exactly copy any of the example agents
 * Do not use for loops, use jnp vector operations instead
 * Instead of
 ```python
 if array < ...:
     a = 1
 else:
     a = 2
 ```
 do
 ```python
 a = jnp.where(a < ..., 1, 2)
 ```
"""

FIX_SYSTEM_PROPMPT = f"""\
We are going to write a competitive agent that controls boids to swarm against an opponents boids.

Example agent 1:
```python
{read_agent('random')}
```

Example agent 2:
```python
{read_agent('simple')}
```

Return your whole agent as **concise** & **compact** standalone module.

You will need imports, for example:
```python
import jax
import jax.numpy as jnp

from swarm.env import State
```

And to implement this function (you can use jax.jit on functions that act calls)
```python
def act(state: State, team: int, key: jax.random.PRNGKey) -> tuple[jnp.ndarray, jnp.ndarray]:
    ...
```

Do not exactly copy any of the example agents.
"""

def evaluate(src):
    write_src(src, "vibe")
    print("[tournament] Starting tournament...")
    results = tournament.run(num_rounds_per_matchup=2, episode_length=128)
    return [
        {"name": result["name"], "reward": float(result["reward"])}
        for result in results
    ]

def make_src_request(messages, host, port):
    start = time.perf_counter()
    response = requests.post(
        f"http://{host}:{port}/v1/chat/completions",
        json={
            "messages": messages,
            # "temperature": 1.0,
            "conversation_id": str(uuid.uuid4()),
            "stream": False,
            "cache_prompt": True,
            "max_tokens": 16384,
        },
    )
    print(f"[request] Duration {time.perf_counter() - start:.2f}...")
    data = response.json()
    print(data)
    choices = data["choices"]
    choice = choices[0]
    message = choice["message"]
    content = message["content"]
    # content = choice["text"]
    try:
        src = content.split("```python")[1]
        src = src.split("```")[0]
    except IndexError as e:
        print(content)
        raise e
    return src

def vibe(host, port):
    print("[vibing] Starting to vibe...")
    return make_src_request([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": VIBE_SYSTEM_PROPMPT},
        {"role": "assistant", "content": "Okay, I'm ready to implement a competitive, novel boid like agent using my knowledge of multi agent swarms, geometry, programming, vector maths, broadcasting rules and jax."},
        {"role": "user", "content": "Implement an innovative and competitive strategy."}
    ], host, port)

def run(host, port):
    run_id = str(uuid.uuid1())
    fixes = 0
    vibes = 0
    errors = 0
    evaluations = 0
    while True:
        src = vibe(host, port)
        vibes += 1
        attempt = 0
        try:
            results = evaluate(src)
            evaluations += 1
            error = None
        except Exception as e:
            stack_trace_lines = [
                line.strip() for line in traceback.format_exc().split("\n")
                if "^^" not in line and not line.startswith("For simplicity, JAX") and not line.startswith("-----")
            ]
            error = "\n".join(stack_trace_lines[-10:])
            errors += 1
            results = None
            print(f"WARNING ERROR: {error}")
        
        log = {
            "attempt": attempt,
            "fixes": fixes,
            "vibes": vibes,
            "errors": errors,
            "evaluations": evaluations,
            "run_id": run_id,
            "error": error,
            "results": results,
            "src": src,
        }
        with open("log.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")
