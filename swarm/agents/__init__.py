from importlib import reload, import_module
import pkgutil

def reload_module(path):
    return reload(import_module(path))

def load_agents():
    agents_module = import_module("swarm.agents")
    return [
        reload_module(f"swarm.agents.{info.name}")
        for info in pkgutil.iter_modules(agents_module.__path__)
    ]

def get_agent(name):
    return reload_module(f"swarm.agents.{name}")
