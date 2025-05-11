import json
import os
import traceback

from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig, SFTConfig, SFTTrainer

from swarm.agents import get_agent_names
from swarm.vibe import evaluate

def read_src(file_path):
    with open(file_path, "r") as f:
        return f.read()

def create_prompt(agent_description):
    return f"""\
Random agent:
```python
{read_src('swarm/agents/random.py')}
```
Centralising agent:
```python
{read_src('swarm/agents/center.py')}
```
{agent_description} agent:
```python
"""

def create_completion(agent_path):
    agent_src = read_src(agent_path)
    return f"""\
{agent_src}
```"""

def reward(content):
    info = {"content": content}
    if '```' not in content:
        info["failure"] = 'NO_CODE_BLOCK'
        return -2.0, -1.0, info
    src, *rest = content.split("```")
    info["rest"] = rest

    try:
        results = evaluate(src)
        matches = [result for result in results if result["name"] == "vibe"]
        r = matches[0]["reward"]
        info["failure"] = None
        return r, r, info
    except SyntaxError as e:
        trace = traceback.format_exc()
        info['trace'] = trace
        info['exception'] = str(type(e))
        info["failure"] = 'PYTHON_EXCEPTION'
        return -1.9, -1.0, info
    except Exception as e:
        trace = traceback.format_exc()
        info['trace'] = trace
        info['exception'] = str(type(e))
        info["failure"] = 'PYTHON_EXCEPTION'
        # TODO: Handle different failures (e.g. doesnt compile vs OOM)
        return -1.5, -1.0, info

def sft(run_id):
    dataset = Dataset.from_list(
        [
            {
                "prompt": create_prompt(agent_name.replace("_", " ").title()),
                "completion": create_completion(f'swarm/agents/{agent_name}.py'),
            }
            for agent_name in get_agent_names()
            if agent_name not in {"vibe"}
        ] + [
            {
                "prompt": create_prompt("Unknown Strategy"),
                "completion": create_completion(f'results/archive/{agent_file}'),
            }
            for agent_file in os.listdir("results/archive")
        ]
    )
    dataset = dataset.train_test_split(test_size=2, seed=1)

    model_name = "Qwen/Qwen3-0.6B"
    trainer = SFTTrainer(
        args=SFTConfig(
            output_dir=f"sft/{run_id}",
            max_seq_length=2048,
            learning_rate=0.0003,
            num_train_epochs=3,
            per_device_eval_batch_size=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            model_init_kwargs={"torch_dtype": "float16"},
            # Evaluation
            eval_on_start=True,
            eval_steps=1,
            eval_strategy="epoch",
            # Checkpointing
            save_strategy="epoch",
            save_steps=1,
        ),
        model=model_name,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=LoraConfig(
            target_modules="all-linear",
            task_type=TaskType.CAUSAL_LM,
        )
    )
    trainer.train()
    trainer.save_model(f"sft/{run_id}/best")
    return f"sft/{run_id}/best"

# def evaluate_model(model, tokenizer):
#     prompt = create_prompt("Unknown Strategy")
#     inputs = tokenizer(prompt, return_tensors="pt")

#     with torch.no_grad():
#         completions = model.generate(
#             **{
#                 key: value.cuda() if isinstance(value, torch.Tensor) else value
#                 for key, value in inputs.items()
#             },
#             max_new_tokens=1024,
#             num_return_sequences=8,
#             eos_token_id=[151645],
#         )

#     for completion in completions:
#         print("*" * 120)
#         output = tokenizer.decode(completion)
#         response = output[len(prompt): ]
#         shaped, r, info = reward(response)
#         print(info)
#         print(shaped, r)

class RewardFunc:
    __name__ = "SwarmReward"

    def __init__(self, run_id):
        self.run_id = run_id

    def __call__(self, completions, *args, **kwargs):
        rewards = []
        infos = []
        for completion in completions:
            shaped, r, info = reward(completion)
            print(shaped, r, info.get("failure"))
            rewards.append(shaped)
            infos.append({
                "reward": r,
                "shaped_reward": shaped,
                "info": info,
            })
        with open("grpo.jsonl", "a") as f:
            f.write(json.dumps({
                "infos": infos,
                "run_id": self.run_id,
            }) + "\n")
        return rewards

def run():
    import uuid
    run_id = str(uuid.uuid1())
    # sft_checkpoint = sft(run_id)
    sft_checkpoint = "sft/e60a3266-2dcf-11f0-a0c7-f02f748483e5/best"

    dataset = Dataset.from_list([
        {"prompt": create_prompt("Unknown Strategy")},
        {"prompt": create_prompt("Asymmetric Costs")},
        {"prompt": create_prompt("Centre of Gravity")},
        {"prompt": create_prompt("Culminating Point")},
        {"prompt": create_prompt("Decisive Point")},
        {"prompt": create_prompt("Fog of War")},
        {"prompt": create_prompt("OODA Loop")},
        {"prompt": create_prompt("Positive Ends")},
        {"prompt": create_prompt("Primary Trinity")},
        {"prompt": create_prompt("Offensive")},
        {"prompt": create_prompt("Overwhelming Mass")},
        {"prompt": create_prompt("Maneuver")},
        {"prompt": create_prompt("Unity of Command")},
        {"prompt": create_prompt("Surprise")},
        {"prompt": create_prompt("Simplicity")},
        {"prompt": create_prompt("Tipping Point")},
        {"prompt": create_prompt("Volatility Uncertainty Complexity and Ambiguity")},
        {"prompt": create_prompt("Choke Point")},
        {"prompt": create_prompt("Defence in Depth")},
        {"prompt": create_prompt("Elastic Defence")},
        {"prompt": create_prompt("Fabian Strategy")},
        {"prompt": create_prompt("Turtling")},
        {"prompt": create_prompt("Bait and Bleed")},
        {"prompt": create_prompt("Blitzkrieg ")},
        {"prompt": create_prompt("Encirclement")},
        {"prompt": create_prompt("Flanking Maneuver")},
        {"prompt": create_prompt("Limited Warfare")},
        {"prompt": create_prompt("Shock and Awe")},        
    ])
    
    model = AutoModelForCausalLM.from_pretrained(sft_checkpoint)
    model.enable_input_require_grads()
    
    trainer = GRPOTrainer(
        args=GRPOConfig(
            output_dir=f"grpo/{run_id}",
            num_train_epochs=1024,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_generations=8,
            max_completion_length=512 + 256,
        ),
        model=model,
        reward_funcs=RewardFunc(run_id),
        train_dataset=dataset,
    )
    trainer.train()
