from collections import deque
import json
import os
import uuid
import types

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    
    batch = 2
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

def reward_formatting(completion, *args, **kwargs):
    """Completion should be formatted."""
    blocks = _parse_blocks(completion)
    return -1.0 if blocks is None else 1.0

def reward_computation_graph(completion, *args, **kwargs):
    """Completions should connect inputs to outputs."""
    reward = -1.0
    G = _create_compute_graph(completion)
    if G is not None:
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
        outputs = ["dvx", "dvy"]
        possible_paths = len(input_params) * len(outputs)
        path_value = (1 / possible_paths) * 2
        for input_param in input_params:
            for output in outputs:
                if input_param not in G.nodes:
                    continue
                if output not in G.nodes:
                    continue
                if nx.has_path(G, input_param, output):
                    reward += path_value
    return reward

def reward_tournament(completion, run_id=None, step=None, index=None, metadata=None):
    """Performance against opponents."""
    try:
        agent = _load_agent(completion)
        results = tournament.run(
            [agent],
            OPPONENTS,
            num_rounds_per_matchup=32,
            episode_length=128,
        )
        matches = [
            result for result in results
            if result["name"] == "tmp_agent"
        ]
        reward =float(matches[0]["reward"])
    except Exception as e:
        reward = -1.0

    if run_id is not None:
        # Persist completions and reward (maybe SFT later?)
        # N.B. This doesnt seem like the right place to do this, but I can't
        # find a TrainerCallback that has this context.
        sample_directory = f"results/logdir/{run_id}/completions/{step}/{index}"
        os.makedirs(sample_directory, exist_ok=True)
        with open(f"{sample_directory}/completion.txt", "w") as f:
            f.write(completion)
        with open(f"results/logdir/rewards.jsonl", "a") as f:
            data = {
                "reward": reward,
                "run_id": run_id,
                "step": step,
                "index": index,
                "metadata": metadata,
            }
            f.write(json.dumps(data) + "\n")
    
    return reward

def make_reward_fn(func, run_id):
    step = 0
    def batched_reward(prompts, completions, *kwargs):
        nonlocal step
        step += 1
        return [
            func(completion, run_id, step, index)
            for index, completion in enumerate(completions)
        ]
    batched_reward.__name__ = func.__name__
    return batched_reward

def run():
    run_id = str(uuid.uuid1()).split("-")[0]
    print(f"{run_id=}")

    # --------------------------------- Model ---------------------------------
    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    
    # --------------------------------- Lora ----------------------------------
    lora_rank = 32
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
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ------------------------------- Tokenizer -------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------- Rewards --------------------------------
    weighted_reward_fns = [
        (make_reward_fn(reward_tournament, run_id), 1.0),
        (make_reward_fn(reward_computation_graph, run_id), 0.4),
        (make_reward_fn(reward_formatting, run_id), 0.4),
    ]
    reward_weights = [weight for reward_fn, weight in weighted_reward_fns]
    reward_funcs = [reward_fn for reward_fn, weight in weighted_reward_fns]
    
    # --------------------------------- GRPO ----------------------------------
    max_seq_length = 1024
    grpo_prompt = QWEN_PROMPT_TEMPLATE.format(prompt=PROMPT)
    grpo_prompt_length = len(tokenizer(grpo_prompt)["input_ids"])
    grpo_args = GRPOConfig(
        # Disable reference model (save memory)
        beta=0.0,
        # Learning rate
        lr_scheduler_type="cosine",
        learning_rate = 1e-6,
        warmup_ratio=0.1,
        # Optimizer
        optim="paged_adamw_8bit",
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        max_grad_norm=0.1,
        # Batch size
        per_device_train_batch_size=1,
        gradient_accumulation_steps=24,
        num_generations=24,
        # Sequence length
        max_prompt_length=grpo_prompt_length,
        max_completion_length=max_seq_length - grpo_prompt_length,
        num_train_epochs=256,
        reward_weights=reward_weights,
        # Logging
        logging_dir=f"results/logdir/{run_id}",
        logging_strategy="epoch",
        logging_steps=1,
        report_to="tensorboard",
        # Checkpointing
        output_dir=f"results/checkpoints/grpo/{run_id}",
        save_strategy="epoch",
        save_steps=8,
        # Generation sampling
        # https://huggingface.co/Qwen/Qwen3-0.6B#best-practices
        # Non-thinking: Temperature=0.7, TopP=0.8, TopK=20, and MinP=0
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
    )
    grpo_trainer = GRPOTrainer(
        args=grpo_args,
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        train_dataset=Dataset.from_list([{"prompt": grpo_prompt}]),
    )
    grpo_trainer.train()
