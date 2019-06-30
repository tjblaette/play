import world
import network
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import time
import os
import argparse


def timestamp():
    """
    Get the current timestamp.

    Returns:
        Timestamp in seconds (int).
    """
    timestamp = time.time()
    return int(timestamp)

def lineplot(x, y, title, filename):
    """
    Plot a seaborn line plot and save it to file.

    Args:
        x ([numeric]): x-axis coordinates of the data to plot.
        y ([numeric]): y-axis coordinates of the data to plot.
        title (str): Plot title.
        filename (str): File to save the plot to.
    """
    fig,axn = plt.subplots(1,1)
    line_plot = sns.lineplot(x=x, y=y)
    line_plot.set_title(title)
    line_plot.set(ylim=(0,1))
    fig = line_plot.get_figure()
    fig.savefig(filename)


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
            "--epochs",
            help="Number of training epochs to train for.",
            type=int,
            default=1000)
    parser.add_argument(
            "--trainings",
            help="Number of training iterations per epoch.",
            type=int,
            default=10)
    parser.add_argument(
            "--batch",
            help=("Number of game transitions to train on per "
                    "training iteration."),
            type=int,
            default=64)
    parser.add_argument(
            "--gamma",
            help="Discount for expected future rewards in Q-function.",
            type=float,
            default=0.9)
    parser.add_argument(
            "--explore",
            help=("Minimum probability of pursuing exploration strategy "
                    "instead of exploration -> random action instead of "
                    "model-based optimal action"),
            type=float,
            default=0.1)
    parser.add_argument(
            "--prefix",
            help="File index to prefix output folder.",
            type=str,
            default=50)
    parser.add_argument(
            "--suffix",
            help="Descriptive suffix for output folder.",
            type=str,
            default="w-replay-buffer-current-q")
    args = parser.parse_args()

    #######################################
    # SETUP WORLD
    dim = tuple((args.height, args.width))
    game_world = world.World(dim)

    #######################################
    # COLLECT DATA TO TRAIN AI
    epochs = args.epochs
    training_iterations = args.trainings
    batch_size = args.batch
    gamma_decay = args.gamma
    min_exploration_prob = args.explore
    file_index = args.prefix
    suffix =  args.suffix

    network_dir = '_'.join(
        [str(x) for x in [
            file_index,
            dim[0],
            'x',
            dim[1],
            epochs,
            'x',
            training_iterations,
            'x',
            batch_size,
            gamma_decay,
            suffix,
            timestamp()]])

    net = network.Network(np.prod(dim), game_world.snake.ACTION_DIM, network_dir)
    sensitivities = []
    exploration_probs = []
    replay_buffer = []

    for epoch in range(epochs):
        print("---------")
        print("Epoch {}".format(epoch))
        collected_data = False

        while not collected_data or len(replay_buffer) < batch_size:
            exploration_prob = max(min_exploration_prob, 1 - epoch/epochs)
            transitions = world.collect_training_data(
                dim,
                net,
                1,
                gamma_decay,
                exploration_prob)

            replay_buffer += [(transition, epoch)
                for transition in transitions]
            collected_data = True
        print("Replay buffer size: {}".format(len(replay_buffer)))

        #######################################
        # TEST IN SIMULATION
        print("-----")
        tested_worlds = world.play_to_test(
            net,
            game_world.dim,
            exploration_prob=0,
            verbose=False)

        sens = world.sensitivity(tested_worlds)
        print("SENSITIVITY: {}".format(sens))

        world.plot_failures(
            tested_worlds,
            net.checkpoint_dir
                + os.path.sep
                + str(epoch)
                + "_failure-maps_network.png")
        time.sleep(2)

        #######################################
        # TRAIN ON THAT DATA
        print("TRAIN")
        game_world.plot_q_table(
            net,
            filename="q_{}.png".format(network_dir + str(epoch)))

        for _ in range(training_iterations):
            training_data_index = np.random.choice(
                len(replay_buffer),
                min(batch_size, len(replay_buffer)),
                replace=False)
            training_data = [replay_buffer[index] for index in training_data_index]
            training_transitions, ep = zip(*training_data)

            training_true_q = [world.calc_true_exp_reward(net, transition, gamma_decay)
                for transition in training_transitions]
            net.train(training_transitions, training_true_q)

        game_world.plot_q_table(
            net,
            filename="q_{}.png".format(network_dir + str(epoch + 1)))

        #######################################
        # COLLECT STATS
        sensitivities.append(sens)
        exploration_probs.append(exploration_prob)


    #######################################
    # TEST IN SIMULATION
    print("-----")
    tested_worlds = world.play_to_test(
        net,
        game_world.dim,
        exploration_prob=0,
        verbose=False)

    sens = world.sensitivity(tested_worlds)
    sensitivities.append(sens)
    print("SENSITIVITY: {}".format(sens))

    world.plot_failures(
        tested_worlds,
        net.checkpoint_dir + os.path.sep + "failure-maps_network.png")


    #######################################
    # SAVE STATS TO FILE
    pd.DataFrame({network_dir: sensitivities}).to_csv(
        net.checkpoint_dir + os.path.sep + "sensitivity.tsv", sep='\t')
    lineplot(
        np.arange(len(sensitivities)),
        sensitivities,
        "Sensitivity",
        net.checkpoint_dir + os.path.sep + "sensitivity.png")
    lineplot(
        np.arange(len(exploration_probs)),
        exploration_probs,
        "Probability of Exploration per Epoch",
        net.checkpoint_dir + os.path.sep + "exploration-prob.png")

    #######################################
    # TEST RANDOM CONTROL FOR COMPARISON
    tested_worlds = world.play_to_test(
        net,
        game_world.dim,
        exploration_prob=1,
        verbose=False)
    sens = world.sensitivity(tested_worlds)
    print("RANDOM CONTROL: {}".format(sens))
    world.plot_failures(
        tested_worlds,
        net.checkpoint_dir + os.path.sep + "failure-maps_control.png")
    print("-----")

if __name__ == "__main__":
    main()


