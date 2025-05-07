# Swarm Intelligence Simulation

A JAX-based simulation environment for studying swarm intelligence and multi-agent behavior. Agents compete in a 2D environment, demonstrating various strategies from simple boids to complex combat coordination. Agents have been implemented by the vibes (Cursor).

## Environment
- 2D continuous space with multiple agents on each team
- Team-based combat with health and damage mechanics

## Example matches

|  | <img width="125px" height="0px"> | <img width="125px" height="0px"> | <img width="125px" height="0px"> | <img width="125px" height="0px"> | <img width="125px" height="0px"> |
| --- | --- | --- | --- | --- | --- |
|  | health_swarm (0.80) | predator_swarm (0.66) | vortex_swarm (0.31) | fortress_swarm (-0.92) | random (-0.92) |
health_swarm | ![](results/animations/health_swarm_vs_health_swarm.gif) | ![](results/animations/health_swarm_vs_predator_swarm.gif) | ![](results/animations/health_swarm_vs_vortex_swarm.gif) | ![](results/animations/health_swarm_vs_fortress_swarm.gif) | ![](results/animations/health_swarm_vs_random.gif)
predator_swarm | ![](results/animations/predator_swarm_vs_health_swarm.gif) | ![](results/animations/predator_swarm_vs_predator_swarm.gif) | ![](results/animations/predator_swarm_vs_vortex_swarm.gif) | ![](results/animations/predator_swarm_vs_fortress_swarm.gif) | ![](results/animations/predator_swarm_vs_random.gif)
vortex_swarm | ![](results/animations/vortex_swarm_vs_health_swarm.gif) | ![](results/animations/vortex_swarm_vs_predator_swarm.gif) | ![](results/animations/vortex_swarm_vs_vortex_swarm.gif) | ![](results/animations/vortex_swarm_vs_fortress_swarm.gif) | ![](results/animations/vortex_swarm_vs_random.gif)
fortress_swarm | ![](results/animations/fortress_swarm_vs_health_swarm.gif) | ![](results/animations/fortress_swarm_vs_predator_swarm.gif) | ![](results/animations/fortress_swarm_vs_vortex_swarm.gif) | ![](results/animations/fortress_swarm_vs_fortress_swarm.gif) | ![](results/animations/fortress_swarm_vs_random.gif)
random | ![](results/animations/random_vs_health_swarm.gif) | ![](results/animations/random_vs_predator_swarm.gif) | ![](results/animations/random_vs_vortex_swarm.gif) | ![](results/animations/random_vs_fortress_swarm.gif) | ![](results/animations/random_vs_random.gif)
## Average Agent Rewards
```
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


## Usage

1. **Run a Round Robin Tournament**
```bash
python -m -O swarm.run
```

2. **Optimize Configurable Agent**
```bash
python -m -O swarm.optimize
```

3. **Create Animation**
```bash
python -m -O swarm.animate <agent1> <agent2>
```

## Dependencies
- JAX
- NumPy
- Matplotlib (for visualization)
- (optional) bayesian-optimization (for tuning config_swarm)

## Development
- New agent types can be added by implementing the `act(state, team, key) -> tuple[dvx, dvy]` interface
- Evaluation runs matches against multiple opponent types
