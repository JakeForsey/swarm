from collections import deque
import hashlib
import json
import os
import uuid
import types
import math

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOTrainer, GRPOConfig

import jax
import jax.numpy as jnp
import jax.extend.core as jexcore
import networkx as nx

from swarm import tournament
from swarm.agents import get_agent

QWEN_PROMPT_TEMPLATE = """<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>"""

PROMPT = f"""\
Please develop a competitive agent in python (using jax) that competes in a two player environment.

The Rules:
 * The 2D (1.0 x 1.0) map wraps around at the edges
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
    # Time step [batch_size] 
    t: jnp.array,
    # Jax random key
    key: jnp.array, 
    # Positions, velocities and health [batch_size, num_agents]
    ally_x: jnp.array,
    ally_y: jnp.array,
    ally_vx: jnp.array,
    ally_vy: jnp.array,
    ally_health: jnp.array,
    enemy_y: jnp.array,
    enemy_x: jnp.array,
    enemy_vx: jnp.array,
    enemy_vy: jnp.array,
    enemy_health: jnp.array,
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

# ---------------------------------- Parsing ----------------------------------

def _parse_blocks(completion):
    if "```python" not in completion:
        return None
    prefix, src, *other = completion.split("```python")
    if "```" not in src:
        return None
    src, *extra = src.split("```")
    return prefix, src, "```".join(extra) + "```python".join(other)

def _parse_src(completion):
    blocks = _parse_blocks(completion)
    if blocks is None:
        return None
    src = blocks[1]
    return src

def _load_agent(completion):
    src = _parse_src(completion)
    if src is None:
        return None
    try:
        agent = types.ModuleType("tmp_agent")
        exec(src, agent.__dict__)
    except:
        return None
    return agent

def _load_act(completion):
    agent = _load_agent(completion)
    if agent is None:
        return None
    return getattr(agent, "act", None)

def _create_compute_graph(completion):
    act = _load_act(completion)
    if act is None:
        return None
    
    arg_names = [
        't',
        'key',
        'ally_x',
        'ally_y',
        'ally_vx',
        'ally_vy',
        'ally_health',
        'enemy_y',
        'enemy_x',
        'enemy_vx',
        'enemy_vy',
        'enemy_health',
    ]
    
    batch = 8
    num_agents = 32
    shape = (batch, num_agents)

    t = jnp.ones((batch, ))
    key = jax.random.PRNGKey(1)
    ally_x = jnp.ones(shape)
    ally_y = jnp.ones(shape)
    ally_vx = jnp.ones(shape)
    ally_vy = jnp.ones(shape)
    ally_health = jnp.ones(shape)
    enemy_y = jnp.ones(shape)
    enemy_x = jnp.ones(shape)
    enemy_vx = jnp.ones(shape)
    enemy_vy = jnp.ones(shape)
    enemy_health = jnp.ones(shape)

    try:
        jaxpr = jax.make_jaxpr(act._fun)(
            t,
            key,
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
        ).jaxpr
    except Exception as e:
        print(e)
        return None
    
    # Build graph
    G = nx.DiGraph()
    name_map = {var: name for var, name in zip(jaxpr.invars, arg_names)}

    used_vars = set()
    counter = 0

    # Map out which variables come from which equation
    produced_by = {}
    for eqn in jaxpr.eqns:
        for out in eqn.outvars:
            produced_by[out] = eqn

    # BFS queue
    queue = deque(jaxpr.outvars)
    for outv, out_name in zip(jaxpr.outvars, ["dvx", "dvy"]):
        name_map[outv] = out_name
        G.add_node(name_map[outv], label=name_map[outv], type="output")
        counter += 1

    while queue:
        var = queue.popleft()
        if var in used_vars:
            continue
        used_vars.add(var)

        # Ensure var has a node
        if var not in name_map:
            name_map[var] = f"v{counter}"
            G.add_node(name_map[var], label=name_map[var], type="var")
            counter += 1

        # If var has no producing equation, it's an input or literal
        if var not in produced_by:
            continue

        eqn = produced_by[var]
        op_node = f"{eqn.primitive.name}_{counter}"
        G.add_node(op_node, label=eqn.primitive.name, type='op')
        counter += 1

        # Connect op → output var
        for outv in eqn.outvars:
            outv_name = name_map.setdefault(outv, f"v{counter}")
            G.add_node(outv_name, label=outv_name, type="var")
            G.add_edge(op_node, outv_name)
            # G.add_edge(outv_name, op_node)
            counter += 1

        # Connect each input var → op
        for invar in eqn.invars:
            if isinstance(invar, jexcore.Literal):
                continue
            if invar not in name_map:
                name_map[invar] = f"v{counter}"
                G.add_node(name_map[invar], label=name_map[invar], type='var')
                counter += 1
            G.add_edge(name_map[invar], op_node)
            # G.add_edge(op_node, name_map[invar])
            queue.append(invar)  # <-- Enqueue recursively

    return G

# ---------------------------------- Rewards ----------------------------------

def reward_prefix_length(completion):
    """Prefix should be 0 characters."""
    blocks = _parse_blocks(completion)
    if blocks is None:
        return -1.0
    prefix, src, suffix = blocks
    prefix_length = len(prefix)
    return -math.log(prefix_length) / 10 if prefix_length else 0

def reward_suffix_length(completion):
    """Suffix should be 0 characters."""
    blocks = _parse_blocks(completion)
    if blocks is None:
        return -1.0
    prefix, src, suffix = blocks
    suffix_length = len(suffix)
    return -math.log(suffix_length) / 10 if suffix_length else 0

def reward_formatting(completion):
    """Completion should be formatted."""
    blocks = _parse_blocks(completion)
    return -1.0 if blocks is None else 1.0

def reward_valid_module(completion):
    """The completion should contain a valid python module."""
    agent_module = _load_agent(completion)
    return -1.0 if agent_module is None else 1.0

def reward_has_act_function(completion):
    """The completion should contain a valid `act` function."""
    act = _load_act(completion)
    return -1.0 if act is None else 1.0

def reward_no_for_loops(completion):
    """Shouldnt use for loops."""
    return -(completion.count("for ") * 0.1)

def reward_no_if_statements(completion):
    """Shouldnt use if statements."""
    return -(completion.count("if ") * 0.1)

def reward_no_doc_string(completion):
    """Shouldnt write doc strings."""
    doc_string_count = completion.count('"""') // 2
    return -(doc_string_count * 0.1)

def reward_no_import_numpy(completion):
    """Shouldnt import numpy."""
    return -(completion.count('import numpy') * 0.1)

def reward_no_import_jaxtyping(completion):
    """Shouldnt import jaxtyping"""
    return -(completion.count('jaxtyping') * 0.1)

def reward_no_print(completion):
    """Shouldnt print anything."""
    return -(completion.count('print(') * 0.1)

def reward_computation_graph(completion):
    G = _create_compute_graph(completion)
    if G is None:
        return -1.0
    reward = 0
    input_params = [
        "ally_x",
        "ally_y",
        "ally_vx",
        "ally_vy",
        "ally_health",
        "enemy_y",
        "enemy_x",
        "enemy_vx",
        "enemy_vy",
        "enemy_health",
    ]
    for sources, dest in [
        (input_params, "dvx"),
        (input_params, "dvy"),
    ]:
        for source in sources:
            if source not in G.nodes:
                continue
            if dest not in G.nodes:
                continue
            if nx.has_path(G, source, dest):
                print(source, dest)
                reward += 0.03
    return reward

def reward_fn(completion):
    """Performance against opponents."""
    try:
        agent = _load_agent(completion)
        results = tournament.run(
            [agent],
            OPPONENTS,
            num_rounds_per_matchup=32,
            episode_length=128,
        )
    except Exception as e:
        return -1.0

    matches = [result for result in results if result["name"] == "tmp_agent"]
    reward = float(matches[0]["reward"])
    return reward

AUX_REWARD_FNS = [
    reward_prefix_length,
    reward_suffix_length,
    reward_formatting,
    reward_valid_module,
    reward_has_act_function,
    reward_no_for_loops,
    reward_no_if_statements,
    reward_no_doc_string,
    reward_no_import_numpy,
    reward_no_import_jaxtyping,
    reward_no_print,
    reward_computation_graph,
]

# with open("results/src/3294e8e9/7/-0.887499988079071/d7a7a0be6c284f9cadc19aa26e5c3dbcce9e84dd6e812bee470c731639563d15/src.py", "r") as f:
#     data= f.read()
# completion = f"""
# ```python
# {data}
# ```"""

# print(reward_computation_graph(completion))
# exit(1)

class RewardFunc:
    __name__ = "SwarmReward"

    def __init__(self, run_id):
        self.run_id = run_id
        self.step = 0

    def __call__(self, completions, *args, **kwargs):
        rewards = []
        for completion in completions:
            aux_rewards = {}
            for aux_reward_fn in AUX_REWARD_FNS:
                aux_reward = aux_reward_fn(completion)
                print(aux_reward_fn.__name__, aux_reward)
                if aux_reward is not None:
                    aux_rewards[aux_reward_fn.__name__] = aux_reward
            
            reward = reward_fn(completion)
            shaped = reward + (sum(aux_rewards.values()) / 10)
            
            data = {
                **aux_rewards,
                "reward": reward,
                "reward_shaped": shaped,
                "step": self.step,
                "run_id": self.run_id,
            }
            
            completion_hash = hashlib.sha256(completion.encode()).hexdigest()
            directory = f"results/src/{self.run_id}/{self.step}/{reward}/{completion_hash}"
            
            os.makedirs(directory, exist_ok=True)
            with open(f"{directory}/completion.txt", "w") as f:
                f.write(completion)
            with open(f"{directory}/data.json", "w") as f:
                json.dump(data, f)
            src = _parse_src(completion)
            if src is not None:
                with open(f"{directory}/src.py", "w") as f:
                    f.write(src)

            print(shaped, reward)
            
            rewards.append(shaped)
            with open("grpo2.jsonl", "a") as f:
                f.write(json.dumps(data) + "\n")
            
        self.step += 1
        return rewards

def run():
    run_id = str(uuid.uuid1()).split("-")[0]
    print(run_id)

    # --------------------------------- Model ---------------------------------
    model_name = "Qwen/Qwen3-0.6B"
    # model_name = "Qwen/Qwen3-4B"
    prompt_template = QWEN_PROMPT_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # 4bit not available on 1080 Ti (maybe?)
        # quantization_config=BitsAndBytesConfig(load_in_4bit=True)
    )

    # --------------------------------- Lora -----------------------------------
    lora_rank = 16
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank,
        lora_dropout=0,
        bias="none",
        use_rslora=True,
        # loftq_config=None,
    )
    model = get_peft_model(model, peft_config)

    # ------------------------------- Tokenizer -------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_seq_length = 1024
    prompt = prompt_template.format(prompt=PROMPT)
    prompt_length = len(tokenizer(prompt)["input_ids"])

    # ------------------------------- Dataset ---------------------------------
    dataset = Dataset.from_list([{"prompt": prompt}])

    # --------------------------------- GRPO ----------------------------------
    trainer_args = GRPOConfig(
        learning_rate=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_generations=16,
        max_prompt_length=prompt_length,
        max_completion_length=max_seq_length - prompt_length,
        num_train_epochs=1024,
        max_grad_norm=0.1,
        report_to="none",
        output_dir=f"grpo/{run_id}",
        # https://huggingface.co/Qwen/Qwen3-0.6B#best-practices
        # Non-thinking: Temperature=0.7, TopP=0.8, TopK=20, and MinP=0
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
    )
    trainer = GRPOTrainer(
        args=trainer_args,
        model=model,
        processing_class=tokenizer,
        reward_funcs=RewardFunc(run_id),
        train_dataset=dataset,
    )
    trainer.train()
