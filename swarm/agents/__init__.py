import importlib
import pkgutil

agents_module = importlib.import_module("swarm.agents")

def load_agents():
    return [
        importlib.import_module(f"swarm.agents.{info.name}")
        for info in pkgutil.iter_modules(agents_module.__path__)
    ]

def get_agent(name):
    return importlib.import_module(f"swarm.agents.{name}")
