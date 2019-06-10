import sys
import numpy as np
import snake
import network
import visworld
import random
import time
import os
import pprint
import seaborn as sns
import matplotlib.pyplot as plt
import copy

class World():
    def __init__(self, dim, this_snake=None, foods=None, obstacles=None, should_render=False):
        """
        Initialize the World and a Snake to wander it.

        dim (int, int): Dimension tuple defining the size of the world.
        this_snake (Snake): Snake to wander the World.
        foods: List of coordinate tuples, corresponding to food items
            positioned in the world, which the snake can eat.
        obstacles: List of coordinate tuples, corresponding to
            obstacles in the world, which kill the snake if it
            moves into these. 
            ---  NOT IMPLEMENTED YET!  ----
        should_render (bool): Whether the world should be rendered
            using external module (via pygame).
            ---  NOT IMPLEMENTED YET!  ----
        """
        # define the world
        self.dim = dim
        width, height = self.dim

        # define string representation of the world
        self.EMPTY = ' '
        self.FOOD = 'x'
        self.OBSTACLE = '='

        # define rewards to be given for the snake's actions
        # -> it must be worth getting food across the entire world
        self.FOOD_SCORE = sum(self.dim)
        # -> death must be the least optimal solution 
        self.OBSTACLE_SCORE = -self.FOOD_SCORE
        # -> reward staying alive
        self.EMPTY_SCORE = 1

        # fill the world with content
        self.snake = this_snake
        self.foods = foods
        self.obstacles = obstacles

        if not self.snake:
            self.snake = snake.Snake(self.one_empty_field())
        if self.foods is None:
            self.foods = [self.one_empty_field()]
            #self.foods = []
        if self.obstacles is None:
            self.obstacles = []

        # test whether the world should be rendered using external module
        self.should_render = should_render
        if self.should_render:
            self.vis = visworld.Vis(self.dim, self.BLOCKSIZE)

    def get_map(self):
        """
        Obtain a 2D string-array representation of the world,
        including snake, food and obstacle fields.

        Returns:
            Two dimensional numpy array of strings.
        """
        world_map = np.full(self.dim, self.EMPTY)
        if self.foods:
            for food in self.foods:
                world_map[food] = self.FOOD
        if self.snake:
            for coord, segment in zip(self.snake.pos, self.snake.segments):
                world_map[coord] = segment
        if self.obstacles:
            for obst in self.obstacles:
                world_map[obst] = self.OBSTACLE
        return world_map

    def all_empty_fields(self):
        """
        Collect all of the world's empty fields.

        Returns:
            A list of coordinate tuples (int, int).
        """
        empty = []
        world_map = self.get_map()
        height, width = self.dim
        for i in range(height):
            for j in range(width):
                if world_map[i,j] == self.EMPTY:
                    empty.append((i,j))
        return empty

    # catch filled-up world?!
    def one_empty_field(self):
        """
        Return one random empty field of the world.

        Returns:
            A single coordinate tuple (int, int).
        """
        return random.choice(self.all_empty_fields())

    def get_state(self):
        """
        Obtain an integer representation of the world
        by converting symbols of the respective string
        representation to ASCII code. Use this
        representation as the _state_ of the world
        for network learning.

        Returns:
            A flattened 1D tuple of integers.
        """
        world_map = self.get_map()
        state = world_map.flatten()
        mapping = {
                self.EMPTY : 0,
                self.snake.HEAD : 1,
                self.snake.BODY : 2,
                self.snake.TAIL : 2,
                self.FOOD : -1,
                self.OBSTACLE : -2
                }
        state = [mapping[x] for x in state]
        state = tuple(state)
        return state

    def get_optimal_action(self, net, verbose):
        """
        Using a given neural network, predict
        the optimal action to take in the current state
        of the world. The optimal action is the one 
        that maximizes the expected reward that is
        predicted by the network.

        Args:
            net (Network): Instance of Network,
                which is used to predict the expected
                reward & determine the optimal action.
            verbose (bool): Whether to print expected
                future rewards predicted by the net.

        Returns:
            Optimal action (int).
        """
        current_state = self.get_state()
        expected_rewards = net.predict([current_state], [np.ones(4)], verbose)
        opt_action = np.argmax(expected_rewards)
        return opt_action

    def get_random_action(self):
        """
        Select an action at random.

        Returns:
            A random action (int).
        """
        return random.randrange(0, self.snake.ACTION_DIM)

    def get_next_action(self, network, exploration_prob, verbose):
        """
        Weigh exploration vs exploitation and decide on the next
        action for the current world's state.
        Exploration = Choose a random action.
        Exploitation = Choose optimal action.

        Args:
            network (Network): Used to predict the optimal action.
            exploration_prob (float): Probability of exploration
                vs exploitation, should be [0,1].
            verbose (bool): Whether to print type of action
                selected (exploration vs exploitation) and
                the expected future rewards for the latter.

        Returns:
            An action (int).
        """
        # random.random -> float[0,1)
        if random.random() >= exploration_prob:
            if verbose:
                print("exploiting!")
            assert exploration_prob < 1
            return self.get_optimal_action(network, verbose)
        if verbose:
            print("exploring!")
        assert exploration_prob > 0
        return self.get_random_action()

    def is_snake_out_of_bounds(self, verbose):
        """
        Test whether the snake has moved out of the world.

        Args:
            verbose (bool): Whether to print output.

        Returns:
            Boolean.
        """
        x,y = self.snake.pos[0]
        height, width = self.dim
        if x < 0 or y < 0 or x >= width or y >= height:
            if verbose:
                print("Snake moved out of World")
            return True
        return False

    def is_snake_at_food(self, verbose):
        """
        Test whether the snake has reached food.

        Args:
            verbose (bool): Whether to print output.

        Returns:
            Boolean.
        """
        for food in self.foods:
            if self.snake.pos[0] == food:
                if verbose:
                    print("Snake reached food")
                return True
        return False

    def is_snake_in_obstacle(self, verbose):
        """
        Test whether the snake has moved into an obstacle.

        Args:
            verbose (bool): Whether to print output.

        Returns:
            Boolean.
        """
        if self.snake.pos[0] in self.obstacles:
            if verbose:
                print("Snake ran into obstacle")
            return True
        if len(self.snake.pos) > 1 and self.snake.pos[0] in self.snake.pos[1:]:
            if verbose:
                print("Snake ran into itself")
            return True
        return False

    # adjust for > 1 food items
    def update_foods(self):
        """
        Move food to a new, empty field.
        """
        self.foods = [self.one_empty_field()]

    # adjust for > 1 snake by passing snake as arg
    def update_snake(self, verbose):
        """
        Check consequences of the snake's previous action:
        Kill it, if it moved out of the world.
        Kill it, if it moved into an obstacle.
        Let it eat food, if it reached any.
        In any case, reward it appropriately.

        Args:
            verbose (bool): Whether to print descriptive output.
        """
        if self.is_snake_out_of_bounds(verbose):
            self.snake.reward(self.OBSTACLE_SCORE)
            self.snake.die(verbose)
        elif self.is_snake_at_food(verbose):
            self.snake.reward(self.FOOD_SCORE)
            self.update_foods()
            #self.snake.grow_on_next_move = True
        elif self.is_snake_in_obstacle(verbose):
            self.snake.reward(self.OBSTACLE_SCORE)
            self.snake.die(verbose)
        else:
            self.snake.reward(self.EMPTY_SCORE)

    # properly represent food
    # -> currently, q is only sampled for 1 random food pos
    def get_q_table(self, net):
        """
        For a given world and network, calculate the expected 
        rewards / q-values of all states and actions.

        Args:
            network (Network): Estimates q-values.

        Returns:
            A 3D numpy array with one q-value estimate per 
            possible state and action. The first two dimensions
            define the state, the third dimension the action.
        """
        q_table = []
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                world2 = World(self.dim, this_snake=snake.Snake((i,j)))
                state = world2.get_state()
                q = net.predict([state], [np.ones(world2.snake.ACTION_DIM)])
                q_table += [q]
        q_table = np.array(q_table).reshape(self.dim + (self.snake.ACTION_DIM,))
        return q_table

    def plot_q_table(self, net, filename="snake_q-table.png"):
        """
        For a given world and network, plot the expected
        rewards / q-values of all states and actions as 
        heatmaps. Plot one 2D heatmap per possible action.
        Save all heatmaps to the same file.

        Args:
            network (Network): Estimates q-values.
            filename (str): To save the plots to.
        """
        q_maps = []
        q_table = self.get_q_table(net)
        fig,axn = plt.subplots(2, 2)
        
        for action,ax in zip(self.snake.ACTION_SPACE, axn.flat):
            q_action = q_table[:,:,action]
            sns.heatmap(
                    q_action,
                    ax=ax,
                    vmin=-10,
                    vmax=30,
                    center=0,
                    cmap="RdBu_r",
                    linewidths=.3,
                    cbar=True,
                    square=True)
            ax.set_title(self.snake.ACTION_SPACE_LIT[action])
            ax.set_yticks([])
            ax.set_xticks([])
        plt.savefig(net.checkpoint_dir + os.path.sep + filename)
        plt.close(fig)

    def play_simulation(self, net, exploration_prob=0, verbose=True):
        """
        Simulate an AI-controlled game of Snake. Use exploitation
        strategy only, to always take the (predicted) optimal action.

        Args:
            net (Network): To control the world's snake.
            exploration_prob (float): Probability of performing
                    exploration vs exploitation.
            verbose (bool): Whether to print descriptive output
                    during simulation.
        """
        while self.snake.alive:
            self.visualize(verbose)
            self.move_snake(net, exploration_prob, verbose)
            self.update_snake(verbose)


    def move_snake(self, net, exploration_prob, verbose):
        action = self.get_next_action(net, exploration_prob, verbose)
        self.snake.set_direction(action)
        self.snake.move()
        return action

    def visualize(self, verbose):
        if verbose:
            world_map = self.get_map()
            print("active world:")
            pprint.pprint(world_map)
            time.sleep(2)

def get_transitions(states, actions, rewards):
    """
    collapse sequences of states, actions and rewards
    to individual state transitions / experiences.

    Args:
        states: List of states of the world.
        actions: List of actions taken in these states.
        rewards: List of rewards obtained for these actions.

    Returns:
        List of (state, action, reward, next_state) tuples.
    """
    transitions = []
    for i in range(len(states) - 1):
        transition = (states[i], actions[i], rewards[i], states[i+1])
        transitions.append(transition)
    return transitions

def calc_true_exp_reward(net, transition, gamma_decay):
    """
    Calculate the true expected reward / q-value for
    a given transition.

    Args:
        net (Network): Predicts q of next_state.
        transition (state, action, reward, next_state).
        gamma_decay (float): weight of future rewards.

    Returns:
        A 1D numpy array where array[action] is the
        true q and the entries of all other possible
        actions are 0.
    """
    state, action, reward, next_state = transition
    true_reward = reward + gamma_decay * np.amax(net.predict([next_state], [np.ones(4)]))
    
    reward_vec = np.zeros(4)
    reward_vec[action] = true_reward
    return reward_vec.tolist()


def play_to_train(dim, net, exploration_prob, verbose=False):
    """
    Play the game / simulate a world to collect sequences
    of states, actions and rewards for training. Training
    does _not_ take place here though.

    Args:
        dim (int, int): Dimension of the world to simulate.
        net (Network): Network to predict optimal actions
            for exploitation.
        exploration_prob (float): Probability of exploration
            vs exploitation.

    Returns:
        states: List of observed states of the world.
        actions: List of actions taken in these states.
        rewards: List of rewards obtained for these actions.
    """
    world = World(dim)

    states = []
    actions = []
    rewards = []
    
    if verbose:
        print("Let's play")
    while world.snake.alive and len(actions) < sum(dim):
        world.visualize(verbose)

        state = world.get_state()
        action = world.move_snake(net, exploration_prob, verbose)
        world.update_snake(verbose)
        reward = world.snake.last_reward
        if verbose:
            print("Actual reward: {}".format(reward))

        states.append(state)
        actions.append(action)
        rewards.append(reward)

    return states, actions, rewards

def play_to_test(net, dim, exploration_prob, verbose):
    """
    Simulate snake games to determine whether
    a given network successfully steers the
    snake to the food.

    Args:
        net (Network): To assess and to steer the
            snake in simulation.
        dim (int, int): Dimension of the world
            to simulate.
        exploration_prob (float): Probability
            of performing exploration vs
            exploitation. Use 1 to properly
            assess the network's performance;
            use 0 for an entirely random control.
        verbose (bool): Whether to print descriptive
            output during simulation.

    Returns:
        List of (World, success (bool)) tuples.
    """
    tested_worlds = []

    for _ in range((dim[0] * dim[1]) **2 *3):
        world = World(dim)
        world_at_ini = copy.deepcopy(world)
        moves = 0
        success = 0

        while world.snake.alive and moves < dim[0] + dim[1]:
            world.move_snake(net, exploration_prob, verbose)
            moves += 1

            if world.is_snake_at_food(verbose=False):
                success = 1
                break
            world.update_snake(verbose=False)

        tested_worlds.append((world_at_ini, success))

    return tested_worlds

def sensitivity(tested_worlds):
    """
    Calculate the total fraction of successful
    snake games to assess a model's performance.

    Args:
        tested_worlds: List of (World, success (bool))
            tuples to process.

    Returns:
        Fraction of successful simulations (float of [0,1]).
    """
    successes = [success for world, success in tested_worlds]
    return sum(successes) / len(tested_worlds)

def plot_failures(tested_worlds, filename="failures.png"):
    """
    Plot heatmaps of the per-field fraction of
    snakes that failed to reach a given food
    given a set of simulation results. Save
    heatmaps to file.

    Args:
        tested_worlds: List of (World, success (bool))
            tuples to process.
        filename (str): File to save heatmaps to.
    """
    failed_worlds = [world for world, success in tested_worlds if not success]

    if failed_worlds:
        failed_snakes = np.zeros(failed_worlds[0].dim)
        failed_foods  = np.zeros(failed_worlds[0].dim)

        for world in failed_worlds:
            failed_snakes[world.snake.pos[0]] += 1
            failed_foods[world.foods[0]] += 1

        tested_snakes = np.zeros(tested_worlds[0][0].dim)
        tested_foods  = np.zeros(tested_worlds[0][0].dim)
        for world, success in tested_worlds:
            tested_snakes[world.snake.pos[0]] += 1
            tested_foods[world.foods[0]] += 1

        failed_snakes /= tested_snakes
        failed_foods /= tested_foods

        fig,axn = plt.subplots(1,2)
        for failed, plot_title, ax in zip(
                [failed_snakes, failed_foods],
                ["Fraction of\nfailed snakes", "Fraction of\nfailed foods"],
                axn.flat):
            sns.heatmap(
                    failed,
                    ax=ax,
                    vmin=0,
                    vmax=1,
                    center=0.5,
                    cmap="YlOrRd",
                    linewidths=.3,
                    cbar=True,
                    square=True)
            ax.set_title(plot_title)
            ax.set_yticks([])
            ax.set_xticks([])
        plt.savefig(filename)
        plt.close(fig)



def collect_training_data(dim, net, batch_size, gamma_decay, exploration_prob):
    """
    Simulate games of snake to collect state transitions
    for network training.

    Args:
        dim (int, int): Dimension of the world
            to simulate.
        net (Network): Network to control snake
            in simulation for exploitation strategy.
        batch_size (int): Number of worlds to simulate
            and collect transitions from.
        gamma_decay (float): Discount of expected future
            rewards to weigh immediate vs future rewards.
        exploration_prob (float): Probability of
            performing exploration (random action)
            instead of exploitation (optimal action
            predicted by net).

    Returns:
        List of transitions: [(state, action, rewards, next_state)]
    """
    ep_states = []
    ep_actions = []
    ep_rewards = []

    # get batch of training examples
    for _ in range(batch_size):
        states, actions, rewards = play_to_train(
                dim,
                net,
                exploration_prob=exploration_prob
                )
        ep_states += states
        ep_actions += actions
        ep_rewards += rewards

    transitions = get_transitions(ep_states, ep_actions, ep_rewards)
    print("total transitions: {}".format(len(transitions)))
    transitions = list(set(transitions))
    print("unique transitions: {}".format(len(transitions)))
    random.shuffle(transitions)

    return transitions

def lineplot(x, y, title, filename):
    fig,axn = plt.subplots(1,1)
    line_plot = sns.lineplot(x=x, y=y)
    line_plot.set_title(title)
    fig = line_plot.get_figure()
    fig.savefig(filename)


def main():
    #######################################
    # SETUP WORLD
    dim = (5,5)
    world = World(dim)
    
    #######################################
    # Collect training data by simulation
    epochs = 50
    batch_size = 200
    gamma_decay = 0.9
    file_index = 34
    suffix =  'opt-state-mapping'
    suffix =  'opt-state-mapping-normalized'
    suffix =  'opt-state-mapping-normalized-range-only'
    suffix =  'opt-state-mapping-normalized-range-only2'
    suffix =  'opt-state-mapping-normalized-range-only-sgd'
    suffix =  'test-lineplots'

    network_dir = '_'.join([str(x) for x in [file_index, dim[0], 'x', dim[1], epochs, 'x', batch_size, gamma_decay, suffix]])
    net = network.Network(dim[0]*dim[1], world.snake.ACTION_DIM, network_dir)
    sensitivities = []
    exploration_probs = []

    for epoch in range(epochs):
        print("---------")
        print("Epoch {}".format(epoch))

        exploration_prob = max(0.1, 1 - epoch/epochs)
        transitions = collect_training_data(
                dim,
                net,
                batch_size = batch_size,
                gamma_decay = gamma_decay,
                exploration_prob=exploration_prob
                )

        true_q = [calc_true_exp_reward(net, transition, gamma_decay) for transition in transitions]

        #######################################
        # TEST IN SIMULATION
        print("-----")
        tested_worlds = play_to_test(net, world.dim, exploration_prob=0, verbose=False)
        sens = sensitivity(tested_worlds)
        print("SENSITIVITY: {}".format(sens))
        plot_failures(tested_worlds, net.checkpoint_dir + os.path.sep + str(epoch) + "_failure-maps_network.png")
        time.sleep(2)

        #######################################
        # TRAIN ON THAT DATA
        print("TRAIN")
        world.plot_q_table(net, filename="q_{}.png".format(network_dir + str(epoch)))
        net.train(transitions, true_q)
        world.plot_q_table(net, filename="q_{}.png".format(network_dir + str(epoch + 1)))

        #######################################
        # COLLECT STATS
        sensitivities.append(sens)
        exploration_probs.append(exploration_prob)


    #######################################
    # TEST IN SIMULATION
    print("-----")
    tested_worlds = play_to_test(net, world.dim, exploration_prob=0, verbose=False)
    sens = sensitivity(tested_worlds)
    sensitivities.append(sens)
    print("SENSITIVITY: {}".format(sens))
    plot_failures(tested_worlds, net.checkpoint_dir + os.path.sep + "failure-maps_network.png")

    lineplot(np.arange(len(sensitivities)), sensitivities, "Sensitivity", net.checkpoint_dir + os.path.sep + "sensitivity.png")
    lineplot(np.arange(len(exploration_probs)), exploration_probs, "Probability of Exploration per Epoch", net.checkpoint_dir + os.path.sep + "exploration-prob.png")

    # TEST RANDOM CONTROL FOR COMPARISON
    tested_worlds = play_to_test(net, world.dim, exploration_prob=1, verbose=False)
    sens = sensitivity(tested_worlds)
    print("RANDOM CONTROL: {}".format(sens))
    plot_failures(tested_worlds, net.checkpoint_dir + os.path.sep + "failure-maps_control.png")
    print("-----")
    world.play_simulation(net)

main()
