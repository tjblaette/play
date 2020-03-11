import sys
import numpy as np
import pandas as pd
import snake
import network
import visworld
import random
import time
import os
import pprint
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import copy

class World():
    def __init__(
            self,
            dim,
            this_snake=None,
            foods=None,
            obstacles=None,
            should_render=False):
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
        """
        # define the world
        self.dim = dim
        height, width = self.dim

        # define string representation of the world
        self.EMPTY = ' '
        self.FOOD = 'x'
        self.OBSTACLE = '='

        # define rewards to be given for the snake's actions
        # -> it must be worth getting food across the entire world
        self.FOOD_SCORE = sum(self.dim)
        # -> death must be the least optimal solution
        self.OBSTACLE_SCORE = -self.FOOD_SCORE
        # -> penalize not reaching food
        # -> include both dimensions?
        self.EMPTY_SCORE = -dim[0]/2
        self.WIN_SCORE = 1000

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

        # check whether the world should be rendered using pygame
        self.paused = True
        self.should_render = should_render
        self.vis = visworld.Vis(self.should_render, self.dim)
        self.visualize(verbose=False)

    def copy(self):
        """
        Copy world with novel vis. This is necessary
        because pygame Surface is not pickable and therefore
        cannot be copied directly with copy.deepcopy().

        Returns:
            Copy of self.
        """
        world = World(self.dim)
        world.snake = copy.deepcopy(self.snake)
        world.foods = copy.deepcopy(self.foods)
        world.obstacles = copy.deepcopy(self.obstacles)
        world.vis = visworld.Vis(self.should_render, self.dim)
        return world

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

    def get_string_state(self):
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

    def get_vis_state(self):
        """
        Obtain an integer representation of the world
        using the pygame visualization. Use this
        representation as the _state_ of the world
        for network learning.

        Returns:
            A flattened 1D tuple of integers.
        """
        state = self.vis.get_state()
        state = tuple(state.flatten().tolist())
        return state

    def get_state(self):
        # remember that to change this to string_state,
        # I have to change the networks input dim
        #return self.get_vis_state()
        return self.get_string_state()

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

    def get_next_action(self, network, exploration_prob=0, verbose=False):
        """
        Get the next action that the snake should perform
        """
        if network is None:
            next_action = self.get_next_action_from_user()
        else:
            next_action = self.get_next_action_from_network(
                network, exploration_prob, verbose)

        return next_action


    def get_next_action_from_user(self):
        """
        Allow the user to control the snake:
        User sets the direction that the snake should move in
        via the keyboard. This direction corresponds to the
        action that should be performed.

        Returns:
            An action (int).
        """
        return self.snake.direction


    def get_next_action_from_network(self, network, exploration_prob, verbose):
        """
        Allow the given network to control the snake.
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
        y,x = self.snake.pos[0]
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
        if (self.is_snake_out_of_bounds(verbose) or
                self.is_snake_in_obstacle(verbose) or
                self.snake.moves_since_last_eaten > sum(self.dim) * 2):
            self.snake.reward(self.OBSTACLE_SCORE)
            self.snake.die(verbose)
            if self.should_render:
                self.vis.end()
        elif self.is_snake_at_food(verbose):
            self.snake.reward(self.FOOD_SCORE)
            self.snake.moves_since_last_eaten = 0
            if not self.all_empty_fields():
                self.snake.reward(self.WIN_SCORE)
                self.snake.win(verbose)
            else:
                self.update_foods()
                self.snake.grow_on_next_move = True
        else:
            self.snake.reward(self.EMPTY_SCORE)
            self.snake.moves_since_last_eaten += 1

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
        q_table = np.array(q_table)
        q_table = q_table.reshape(self.dim + (self.snake.ACTION_DIM,))
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

    def play_simulation(self, net, exploration_prob=0, verbose=False):
        """
        Play a game of Snake, controlled by the user or a given network.
        When given a network, use exploitation strategy only, to always
        perform the (predicted) optimal action. For a user-controlled
        game, set net to None.

        Args:
            net (Network): To control the world's snake.
            exploration_prob (float): Probability of performing
                    exploration vs exploitation.
            verbose (bool): Whether to print descriptive output
                    during simulation.
        """
        while self.snake.alive:
            self.visualize(verbose)

            if self.should_render:
                self.vis.clock.tick(self.vis.FPS)
                self.vis.check_for_user_input(self)
            elif verbose:
                time.sleep(2)

            if not self.paused:
                self.move_snake(net, exploration_prob, verbose)
                self.update_snake(verbose)

    def move_snake(self, net, exploration_prob, verbose):
        """
        Move snake to the next field.

        Returns:
            The action (int) that was taken.
        """
        action = self.get_next_action(net, exploration_prob, verbose)
        self.snake.set_direction(action)
        self.snake.move()
        return action

    def visualize(self, verbose):
        """
        Visualize the world, by printing it to stdout
        and / or visualizing it using pygame.

        Args:
            verbose (bool): Whether to print the world to stdout.
        """
        if verbose:
            world_map = self.get_map()
            print("active world:")
            pprint.pprint(world_map)
        self.vis.update(self)

def get_transitions(states, actions, rewards, lethal_final_action):
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

    if lethal_final_action:
        transitions.append((states[-1], actions[-1], rewards[-1], None))
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

    if next_state is None:
        true_reward = reward
    else:
        true_reward = (
            reward
            + gamma_decay * np.amax(net.predict([next_state], [np.ones(4)])))

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

    return states, actions, rewards, not world.snake.alive

# COMMENT
def avrg_dist_in(dim):
    """
    COMMENT
    """
    print(dim)
    dist = []
    for _ in range(np.prod(dim) * 2):
        world = World(dim)
        snake = np.array(world.snake.pos[0])
        food = np.array(world.foods[0])
        dist.append(sum(abs(snake - food)))
    print(sum(dist) / len(dist))

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

    for _ in range(min(1000, (np.prod(dim)) **2 *3)):
        world = World(dim)
        world_at_ini = world.copy()
        moves = 0
        success = 0

        while world.snake.alive and moves < sum(dim):
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
    transitions = []

    # get batch of training examples
    for _ in range(batch_size):
        states, actions, rewards, lethal_final_action = play_to_train(
            dim,
            net,
            exploration_prob)
        transitions += get_transitions(
            states,
            actions,
            rewards,
            lethal_final_action)

    print("total transitions: {}".format(len(transitions)))
    transitions = list(set(transitions))
    print("unique transitions: {}".format(len(transitions)))
    random.shuffle(transitions)

    return transitions

def simulate_only(dim, network_dir):
    """
    Simulate an AI-controlled game of snake.
    """
    world = World(dim, should_render=True)
    net = network.Network(np.prod(dim), world.snake.ACTION_DIM, network_dir)
    world.play_simulation(net)



#avrg_dist_in((2,2))
#avrg_dist_in((3,3))
#avrg_dist_in((4,4))
#avrg_dist_in((5,5))
#avrg_dist_in((10,10))
