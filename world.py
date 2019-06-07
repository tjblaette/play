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
        char_to_int = np.vectorize(lambda x: ord(x))
        ascii_map = char_to_int(world_map)
        state = ascii_map.flatten()
        state -= 32
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
                print(world2.get_map())
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
        q_table = self.get_q_table(net)
        print(q_table)
        q_maps = []
        fig,axn = plt.subplots(2, 2)
        
        for action,ax in zip(self.snake.ACTION_SPACE, axn.flat):
            q_action = q_table[:,:,action]
            sns.heatmap(q_action, ax=ax, vmin=-10, vmax=30, center=0, cmap="RdBu_r", linewidths=.3, cbar=True, square=True)
            ax.set_title(self.snake.ACTION_SPACE_LIT[action])
            ax.set_yticks([])
            ax.set_xticks([])
        plt.savefig(net.checkpoint_dir + os.path.sep + filename)

    def play_simulation(self, net):
        """
        Simulate an AI-controlled game of Snake. Use exploitation
        strategy only, to always take the (predicted) optimal action.

        Args:
            net (Network): To control the world's snake.
        """
        while self.snake.alive:
            world_map = self.get_map()
            print("active world:")
            pprint.pprint(world_map)
            next_action = self.get_next_action(net, exploration_prob=0, verbose=True)

            self.snake.set_direction(next_action)
            self.snake.move()
            self.update_snake(verbose=True)
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


def play_to_train(dim, net, exploration_prob, should_render=True):
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
        should_render (bool): If true, print the world's map
            during simulation, before each simulated action.

    Returns:
        states: List of observed states of the world.
        actions: List of actions taken in these states.
        rewards: List of rewards obtained for these actions.
    """
    world = World(dim)

    states = []
    actions = []
    rewards = []
    
    print("Let's play")
    while world.snake.alive and len(actions) < sum(dim):
        world_map = world.get_map()
        if should_render:
            pprint.pprint(world_map)

        state = world.get_state()
        action = world.get_next_action(net, exploration_prob, verbose=True)
        world.snake.set_direction(action)
        world.snake.move()
        world.update_snake(verbose=True)
        reward = world.snake.last_reward
        print("Actual reward: {}".format(reward))

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        #if should_render:
        #    time.sleep(2)

    return states, actions, rewards

def sensitivity(net, dim, exploration_prob):
    """
    Calculate the sensitivity achieved by a
    given network as the proportion of
    simulations in which the snake successfully
    reaches the food.

    Args:
        net (Network): To assess and to steer the
            snake in simulation.
        dim (int, int): Dimension of the world
            to simulate.

    Returns:
        Proportion of successfully reached foods (float).
    """
    total_tries = (dim[0] * dim[1]) **2
    success = 0

    for _ in range(total_tries):
        world = World(dim)
        moves = 0

        while world.snake.alive and moves <= dim[0] + dim[1]:
            world_map = world.get_map()
            next_action = world.get_next_action(net, exploration_prob, verbose=False)
            world.snake.set_direction(next_action)
            world.snake.move()
            moves += 1

            if world.is_snake_at_food(verbose=False):
                success += 1
                break
            world.update_snake(verbose=False)

    return success / total_tries


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


def main():
    #######################################
    # SETUP WORLD AND NEURAL NETWORK
    dim = (3,3)
    world = World(dim)
    network_dir = '28_fixQ_3x3_noFood_exploreBasedOnEp'
    network_dir = '29_fixQ_3x3_noFood_explore20'
    network_dir = '29_fixQ_3x3_noFood_exploreBasedOnEp_20x30'
    network_dir = '29_fixQ_3x3_noFood_explore20_20x30'
    network_dir = '30_fixQ_3x3_noFood_exploreBasedOnEp_20x30'
    network_dir = '30_fixQ_3x3_noFood_explore20_20x30'
    network_dir = '30_fixQ_3x3_noFood_exploreComplex10_20x30'
    network_dir = '30_fixQ_3x3_noFood_exploreComplex10_20x40'
    network_dir = '30_fixQ_3x3_noFood_exploreComplex10_30x30'
    network_dir = '30_fixQ_3x3_noFood_exploreComplex10_30x40'
    network_dir = '30_fixQ_3x3_noFood_exploreComplex20_30x40'
    network_dir = '30_fixQ_3x3_wFood_exploreComplex10_50x200'
    network_dir = '30_fixQ_3x3_wFood_exploreComplex10_layers50x50x50_ep50x200'
    network_dir = '30_fixQ_3x3_wFood_exploreComplex10_layers50x50x50_ep50x200_tmp'
    network_dir = '31_testAfterComments'
    net = network.Network(dim[0]*dim[1], world.snake.ACTION_DIM, network_dir)
    
    #######################################
    # Collect training data by simulation
    epochs = 50
    batch_size = 200
    gamma_decay = 0.9

    for epoch in range(epochs):
        print("---------")
        print("Epoch {}".format(epoch))

        transitions = collect_training_data(
                dim,
                net,
                batch_size = batch_size,
                gamma_decay = gamma_decay,
                exploration_prob=max(0.1, 1 - epoch/epochs)
                )

        true_q = [calc_true_exp_reward(net, transition, gamma_decay) for transition in transitions]

        #######################################
        # TRAIN ON THAT DATA
        print("TRAIN")
        world.plot_q_table(net, filename="q_{}.png".format(network_dir + str(epoch)))
        net.train(transitions, true_q)
        world.plot_q_table(net, filename="q_{}.png".format(network_dir + str(epoch + 1)))

    #######################################
    # TEST IN SIMULATION
    print("-----")
    print("SENSITIVITY: {}".format(sensitivity(net, world.dim, 0)))
    print("RANDOM CONTROL: {}".format(sensitivity(net, world.dim, 1)))
    print("-----")
    world.play_simulation(net)

main()
