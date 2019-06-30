import world
import argparse


def main():
    #######################################
    # READ IN COMMAND LINE ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "height",
            help="Height of the game world.",
            type=int)
    parser.add_argument(
            "width",
            help="Width of the game world.",
            type=int)
    parser.add_argument(
            "network_dir",
            help=("Directory in which the tensorflow model is saved, "
                    "which shall control the game."),
            type=str)
    args = parser.parse_args()

    #######################################
    # SIMULATE AN AI-CONTROLLED GAME
    dim = tuple((args.height, args.width))
    world.simulate_only(dim, args.network_dir)


if __name__ == "__main__":
    main()

