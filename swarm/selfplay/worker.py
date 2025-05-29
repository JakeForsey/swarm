import os
import hashlib
import requests
import types
from typing import Any

from swarm.agents import get_agent_names, get_agent
from swarm import tournament


PROMPT = """\
Please develop a competitive agent in python (using jax) that competes in a two player environment.

The Rules:
 * There is a 2D map (1.0 x 1.0) that wraps around at the edges
 * Each player gets 32 pieces they control
 * Each piece starts with 1.0 health
 * If pieces from opposing teams collide they both take damage
 * If a piece reaches 0.0 health it dies
 * Pieces that are alive regenerate some health each step
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

PARENT_SRC_TEMPLATE = """Version {version}:
```python
{src}
```
"""


def read_src(agent_name: str) -> str:
    with open(f"swarm/agents/{agent_name}.py", "r") as f:
        src = f.read()
    return src


BASELINE_SCORES = [read_src(agent_name) for agent_name in get_agent_names()]


def request_completion(prompt: str) -> str:
    host = os.getenv("LLAMA_SERVER_HOST")
    """Request a completion for a prompt from a llama-server host."""
    response = requests.post(
        f"http://{host}/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": prompt},
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


def reward_to_score(reward: float) -> float:
    # Jax array to float
    reward = float(reward)
    # Conver from [-1, 1] to [0, 2]
    reward += 1
    # Convert from [0, 2] to [0, 1]
    reward /= 2
    # Flip
    reward = 1 - reward
    return reward


def src_id(src: str) -> str:
    return hashlib.sha256(src.encode()).hexdigest()


def src_to_agent(src: str) -> types.ModuleType:
    agent = types.ModuleType(src_id(src))
    exec(src, agent.__dict__)
    return agent


def parse_src(completion: str) -> str:
    if "```python" not in completion:
        return None
    _, src, *__ = completion.split("```python")
    if "```" not in src:
        return None
    src, *_ = src.split("```")
    return src


def evaluate(
    src: str,
    opponent_srcs: list[str],
    num_rounds_per_matchup: int,
    episode_length: int,
) -> tuple[bool, list[dict[str, float]]]:
    try:
        results = tournament.run(
            [src_to_agent(src)],
            [src_to_agent(opp_src) for opp_src in opponent_srcs]
            + [get_agent("random")],
            num_rounds_per_matchup=num_rounds_per_matchup,
            episode_length=episode_length,
        )
        success = True
        scores = {
            result["name"]: reward_to_score(result["reward"])
            for result in results
            if result["name"] != src_id(src)
        }
    except Exception:
        success = False
        scores = {src_id(opp_src): -0.0 for opp_src in opponent_srcs}
    return success, scores


def mutate(parent_srcs: list[str]) -> str:
    if not parent_srcs:
        prompt = PROMPT
        parent_ids = []
    else:
        parent_formatted = [
            PARENT_SRC_TEMPLATE.format(src=parent_src, version=i)
            for i, parent_src in enumerate(parent_srcs)
        ]
        prompt = "\n".join(
            [
                PROMPT,
                *parent_formatted,
                "Create a new agent inspired by the above versions.",
            ]
        )
        parent_ids = [src_id(parent_src) for parent_src in parent_srcs]

    prompt = prompt + "\n/no_think"
    completion = request_completion(prompt)
    src = parse_src(completion)
    return src, parent_ids


def mutate_and_evaluate(
    parent_srcs: list[str],
    generation_srcs: list[str],
    num_rounds_per_matchup: int,
    episode_length: int,
) -> dict[str, Any]:
    child_src, parent_ids = mutate(parent_srcs)
    success, scores = evaluate(
        child_src,
        generation_srcs,
        num_rounds_per_matchup,
        episode_length,
    )
    _, baseline_scores = evaluate(
        child_src,
        BASELINE_SCORES,
        num_rounds_per_matchup,
        episode_length,
    )
    return {
        "success": success,
        "scores": scores,
        "baseline_scores": baseline_scores,
        "src": child_src,
        "id": src_id(child_src),
        "parent_ids": parent_ids,
    }
