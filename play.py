import world
import argparse


def main():
    #######################################
    # READ IN COMMAND LINE ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "width",
            help="Width of the game world.",
            type=int)
    parser.add_argument(
            "height",
            help="Height of the game world.",
            type=int)
    args = parser.parse_args()

    #######################################
    # PLAY A USER-CONTROLLED GAME
    dim = tuple((args.width, args.height))
    game_world = world.World(dim, should_render=True)
    game_world.play_simulation(None)


if __name__ == "__main__":
    main()
