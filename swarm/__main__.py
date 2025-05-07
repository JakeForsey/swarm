import argparse

from swarm import tournament, animate

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

    args = parser.parse_args()

    if args.command == "tournament":
        tournament.run(args.num_rounds_per_matchup, args.episode_length)
    elif args.command == "animate":
        animate.run(args.agent1, args.agent2, args.episode_length)

if __name__ == "__main__":
    main()
