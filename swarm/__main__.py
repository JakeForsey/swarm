import argparse
import uuid

DEFAULT_EPISODE_LENGTH = 128
DEFAULT_NUM_ROUNDS_PER_MATCHUP = 32

def add_episode_length_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--episode-length",
        type=int,
        default=DEFAULT_EPISODE_LENGTH,
        help=f"Episode length (default: {DEFAULT_EPISODE_LENGTH})"
    )

def add_num_rounds_per_matchup_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-rounds-per-matchup",
        type=int,
        default=DEFAULT_NUM_ROUNDS_PER_MATCHUP,
        help=f"Number of rounds per matchup (default: {DEFAULT_NUM_ROUNDS_PER_MATCHUP})"
    )

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    tournament_parser = subparsers.add_parser("tournament", help="Run a tournament")
    add_num_rounds_per_matchup_argument(tournament_parser)
    add_episode_length_argument(tournament_parser)

    animate_parser = subparsers.add_parser("animate", help="Animate a tournament")
    add_episode_length_argument(animate_parser)
    animate_parser.add_argument("agent1", type=str, default="random")
    animate_parser.add_argument("agent2", type=str, default="random")

    vibevolve_parser = subparsers.add_parser("vibevolve", help="VibEvolve new agents")
    add_num_rounds_per_matchup_argument(vibevolve_parser)
    add_episode_length_argument(vibevolve_parser)
    vibevolve_parser.add_argument(
        "--hosts",
        nargs='+',
        default=["cortex1:8080", "cortex2:8080", "cortex2:8081"],
        help="OpenAI compliant LLM server hosts",
    )
    vibevolve_parser.add_argument(
        "--run-id",
        default=str(uuid.uuid1()).split("-")[0],
        help="To continue a run, provide a run id",
    )
    vibevolve_parser.add_argument("--warmup-steps", default=16, type=int)
    vibevolve_parser.add_argument("--num-steps", default=1024, type=int)

    args = parser.parse_args()

    def init_jax():
        import jax
        jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        # jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    if args.command == "tournament":
        init_jax()
        from swarm import tournament
        results = tournament.run(
            num_rounds_per_matchup=args.num_rounds_per_matchup,
            episode_length=args.episode_length,
        )
        for result in sorted(results, key=lambda x: x["reward"], reverse=True):
            print(f"{result['name']:>20} reward: {result['reward']:.2f}")
    elif args.command == "animate":
        init_jax()
        from swarm import animate
        animate.run(
            agent1_name=args.agent1,
            agent2_name=args.agent2,
            episode_length=args.episode_length,
        )
    elif args.command == "vibevolve":
        init_jax()
        from swarm import vibevolve
        vibevolve.run(
            run_id=args.run_id,
            hosts=args.hosts,
            num_rounds_per_matchup=args.num_rounds_per_matchup,
            episode_length=args.episode_length,
            warmup_steps=args.warmup_steps,
            num_steps=args.num_steps
        )

if __name__ == "__main__":
    main()
