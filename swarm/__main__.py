import argparse

import jax

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

from swarm import tournament, animate, vibe

DEFAULT_EPISODE_LENGTH = 128
DEFAULT_NUM_ROUNDS_PER_MATCHUP = 256

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

    # Create the parser for the "tournament" command
    tournament_parser = subparsers.add_parser("tournament", help="Run a tournament")
    add_num_rounds_per_matchup_argument(tournament_parser)
    add_episode_length_argument(tournament_parser)

    # Create the parser for the "animate" command
    animate_parser = subparsers.add_parser("animate", help="Animate a tournament")
    animate_parser.add_argument("agent1", type=str, default="random")
    animate_parser.add_argument("agent2", type=str, default="random")
    add_episode_length_argument(animate_parser)

    vibe_parser = subparsers.add_parser("vibe", help="Vibe")

    args = parser.parse_args()

    if args.command == "tournament":
        tournament.run(args.num_rounds_per_matchup, args.episode_length)
    elif args.command == "animate":
        animate.run(args.agent1, args.agent2, args.episode_length)
    elif args.command == "vibe":
        vibe.run()

if __name__ == "__main__":
    main()
