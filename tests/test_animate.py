import os

from swarm import animate

def test_animate():
    agent1 = "chaser"
    agent2 = "random"
    path = f"results/animations/{agent1}_vs_{agent2}.gif"
    
    if os.path.exists(path):
        os.unlink(path)
    
    animate.run(agent1, agent2, 4)
    
    assert os.path.exists(path)
    os.unlink(path)
