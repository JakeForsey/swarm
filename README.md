# Swarm Intelligence Simulation

A JAX-based simulation environment for studying swarm intelligence and multi-agent behavior. Agents compete in a 2D environment, demonstrating various strategies from simple boids to complex transformer-based coordination. Agents have been implemented by cursor.

### Environment
- 2D continuous space with multiple agents
- Team-based combat with health and damage mechanics
- Configurable episode length and agent counts

## Usage

1. **Run a Round Robin Tournament**
```bash
python -m swarm.run
```

2. **Optimize Configurable Agent**
```bash
python -m swarm.optimize
```

3. **Create Animation**
```bash
python -m swarm.animate
```

## Dependencies
- JAX
- NumPy
- Matplotlib (for visualization)

## Development
- New agent types can be added by implementing the `act(state, team, key) -> tupl[dvx, dvy]` interface
- Optimization uses evolutionary search to find optimal parameters
- Evaluation runs matches against multiple opponent types
