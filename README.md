# Swarm

A fast (GPU accelerated) two player swarm based environment with a large number of unique agents.

## Usage

### 1. Run a Round Robin Tournament
```bash
python  -m swarm tournament
```

```bash
        health_swarm reward: 0.80
      predator_swarm reward: 0.66
        hunter_swarm reward: 0.61
        config_swarm reward: 0.54
         param_swarm reward: 0.43
        vortex_swarm reward: 0.31
       predator_boid reward: 0.29
         squad_swarm reward: 0.25
        spiral_swarm reward: 0.22
        pincer_swarm reward: 0.19
       concave_swarm reward: 0.11
        center_swarm reward: 0.09
      fortress_swarm reward: -0.08
      tactical_swarm reward: -0.10
         train_swarm reward: -0.11
                boid reward: -0.12
      adaptive_swarm reward: -0.17
          smart_boid reward: -0.32
             fleeing reward: -0.43
          ring_swarm reward: -0.48
            clusters reward: -0.49
        static_swarm reward: -0.60
              chaser reward: -0.66
              random reward: -0.92
```
### 2. Animate a game between two agents
```bash
python -m swarm animate <agent1> <agent2>
```

<img src=results/animations/random_vs_predator_swarm.gif width="250px" height="250px">

See more in [results/animations](results/animations)

### 3. Automatically code agents
```bash
python -m swarm vibe
```

### 4. **Train an LLM to write agents with RL**
```bash
python -m swarm rl
```

## Example matches

| <img width="125px" height="0px"> | <img width="125px" height="0px"> | <img width="125px" height="0px"> |
| --- | --- | --- |
![](results/animations/health_swarm_vs_health_swarm.gif) | ![](results/animations/health_swarm_vs_vortex_swarm.gif) | ![](results/animations/health_swarm_vs_random.gif)
![](results/animations/vortex_swarm_vs_health_swarm.gif) | ![](results/animations/vortex_swarm_vs_vortex_swarm.gif) | ![](results/animations/vortex_swarm_vs_random.gif)
![](results/animations/random_vs_health_swarm.gif) | ![](results/animations/random_vs_vortex_swarm.gif) | ![](results/animations/random_vs_random.gif)
