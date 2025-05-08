import json
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
Write excellent, world class, a JAX environment

The state:
```python
class State(NamedTuple):
    t: jnp.ndarray

    x1: jnp.ndarray
    x2: jnp.ndarray
    y1: jnp.ndarray
    y2: jnp.ndarray

    vx1: jnp.ndarray
    vx2: jnp.ndarray
    vy1: jnp.ndarray
    vy2: jnp.ndarray

    health1: jnp.ndarray
    health2: jnp.ndarray
```

Current best agent:
```python
{read_agent('simple')}
```

Return your whole agent as **concise** & **compact** standalone module that exposes a function
```python
def act(state: State, team: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    ...
```
"""

def evaluate(src):
    write_src(src, "vibe")
    print("[tournament] Starting tournament...")
    results = tournament.run(num_rounds_per_matchup=4, episode_length=32)
    return [
        {"name": result["name"], "reward": float(result["reward"])}
        for result in results
    ]

def make_src_request(message):
    start = time.perf_counter()
    id_slot = 0
    requests.post(f"http://cortex1:8080/slots/{id_slot}?action=erase")
    response = requests.post(
        "http://cortex1:8080/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            "stream": False,
            "cache_prompt": True,
        },
    )
    print(f"[request] Duration {time.perf_counter() - start:.2f}...")
    data = response.json()
    choices = data["choices"]
    choice = choices[0]
    message = choice["message"]
    content = message["content"]
    try:
        src = content.split("```python")[1]
        src = src.split("```")[0]
    except IndexError as e:
        print(content)
        raise e
    return src

def vibe():
    print("[vibing] Starting to vibe...")
    return make_src_request("Write a new agent!")

def fix(src, error):
    print("[fixing] Starting to fix...")
    return make_src_request(f"""\
ERROR:
```bash
{error}
```

Erroring agent code:
```python
{src}
```

Fix the agents code and provide it in full
""")

def run():
    run_id = str(uuid.uuid1())
    fixes = 0
    vibes = 0
    errors = 0
    evaluations = 0
    while True:
        src = vibe()
        vibes += 1
        for attempt in range(3):
            try:
                results = evaluate(src)
                evaluations += 1
                error = None
                # This iteration is done!
                break
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                error = tb[-1]
                errors += 1
                results = None
                print(f"WARNING ERROR: {error}")
                src = fix(src, error)
                fixes += 1
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
