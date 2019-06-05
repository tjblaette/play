import sys
import numpy as np
import snake
import network
import visworld
import random
import time
import pprint
import seaborn as sns
import matplotlib.pyplot as plt

class World():
    def __init__(self, dim, this_snake=None, foods=None, obstacles=None, should_render=False):
        self.dim = dim
        width, height = self.dim

        self.EMPTY = ' '
        self.FOOD = 'x'
        self.OBSTACLE = '='
        self.FOOD_SCORE = sum(self.dim) #it must be worth getting food across the entire world
        self.OBSTACLE_SCORE = -self.FOOD_SCORE #death must be the least optimal solution
        self.EMPTY_SCORE = 1 # reward staying alive

        self.snake = this_snake
        self.foods = foods
        self.obstacles = obstacles

        if not self.snake:
            self.snake = snake.Snake(self.one_empty_field())
        if self.foods is None:
            #self.foods = [self.one_empty_field()]
            self.foods = []
        if self.obstacles is None:
            self.obstacles = []

        self.should_render = should_render
        if self.should_render:
            self.vis = visworld.Vis(self.dim, self.BLOCKSIZE)

    def get_map(self):
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
        empty = []
        world_map = self.get_map()
        height, width = self.dim
        for i in range(height):
            for j in range(width):
                if world_map[i,j] == self.EMPTY:
                    empty.append((i,j))
        return empty

    def one_empty_field(self):
        return random.choice(self.all_empty_fields())

    def get_state(self):
        world_map = self.get_map()
        char_to_int = np.vectorize(lambda x: ord(x))
        ascii_map = char_to_int(world_map)
        state = ascii_map.flatten()
        state -= 32
        return state

    def get_optimal_action(self, net):
        current_state = self.get_state()
        expected_rewards = net.predict([current_state], [np.ones(4)])
        opt_action = np.argmax(expected_rewards)
        return opt_action

    def get_random_action(self):
        return random.randrange(0, self.snake.ACTION_DIM)

    def get_next_action(self, network, exploration_prob=0.1):
        # random.random -> float[0,1)
        if random.random() >= exploration_prob:
            #print("exploiting!")
            assert exploration_prob < 1
            return self.get_optimal_action(network)
        #print("exploring!")
        assert exploration_prob > 0
        return self.get_random_action()

    def is_snake_out_of_bounds(self):
        x,y = self.snake.pos[0]
        height, width = self.dim
        if x < 0 or y < 0 or x >= width or y >= height:
            print("Snake moved out of World")
            return True
        return False

    def is_snake_at_food(self):
        for food in self.foods:
            if self.snake.pos[0] == food:
                print("Snake reached food")
                return True
        return False

    def is_snake_in_obstacle(self):
        if self.snake.pos[0] in self.obstacles:
            print("Snake ran into obstacle")
        if len(self.snake.pos) > 1 and self.snake.pos[0] in self.snake.pos[1:]:
            print("Snake ran into itself")
            return True
        return False

    # adjust for > 1 food items
    def update_foods(self):
        self.foods = [self.one_empty_field()]

    # adjust for > 1 snake by passing snake as arg
    def update_snake(self):
        if self.is_snake_out_of_bounds():
            self.snake.reward(self.OBSTACLE_SCORE)
            self.snake.die()
        elif self.is_snake_at_food():
            self.snake.reward(self.FOOD_SCORE)
            self.update_foods()
            #self.snake.grow_on_next_move = True
        elif self.is_snake_in_obstacle():
            self.snake.reward(self.OBSTACLE_SCORE)
            self.snake.die()
        else:
            self.snake.reward(self.EMPTY_SCORE)

    def get_q_table(self, net):
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
        q_table = self.get_q_table(net)
        print(q_table)
        q_maps = []
        fig,axn = plt.subplots(2, 2)
        
        for action,ax in zip(self.snake.ACTION_SPACE, axn.flat):
            q_action = q_table[:,:,action]
            sns.heatmap(q_action, ax=ax, vmin=-10, vmax=10, center=0, cmap="RdBu_r", linewidths=.3, cbar=True, square=True)
            ax.set_title(self.snake.ACTION_SPACE_LIT[action])
            ax.set_yticks([])
            ax.set_xticks([])
        plt.savefig(filename)


def get_transitions(states, actions, rewards):
    transitions = []
    for i in range(len(states) - 1):
        transition = (states[i], actions[i], rewards[i], states[i+1])
        transitions.append(transition)
    return transitions

def calc_true_exp_reward(net, transition, gamma_decay):
    state, action, reward, next_state = transition
    true_reward = reward + gamma_decay * np.amax(net.predict([next_state], [np.ones(4)]))
    
    reward_vec = np.zeros(4)
    reward_vec[action] = true_reward
    return reward_vec.tolist()

def calc_exp_reward(net, transition):
    state, action, reward, next_state = transition
    action_vec = np.zeros(4)
    action_vec[action]= 1
    exp_rewards = net.predict([state], [action_vec])
    return exp_rewards


def play_to_train(dim, net, exploration_prob=1, should_render=True):
    world = World(dim)

    states = []
    actions = []
    rewards = []
    
    print("Let's play")
    while world.snake.alive and len(actions) < 5:
        world_map = world.get_map()
        if should_render:
            pprint.pprint(world_map)

        state = world.get_state()
        action = world.get_next_action(net, exploration_prob)
        world.snake.set_direction(action)
        world.snake.move()
        world.update_snake()
        reward = world.snake.last_reward
        print("Actual reward: {}".format(reward))

        states.append(state.tolist())
        actions.append(action)
        rewards.append(reward)
        #if should_render:
        #    time.sleep(2)

    return states, actions, rewards



def main():
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
    net = network.Network(dim[0]*dim[1], world.snake.ACTION_DIM, network_dir) # do not hard-code 4

    epochs = 30
    batch_size = 40
    for epoch in range(epochs):
        print("---------")
        print(epoch)
        #print("exploration_prob: {}".format(1- epoch/epochs))

        ep_states = []
        ep_actions = []
        ep_rewards = []

        # get batch of training examples
        for _ in range(batch_size):
            states, actions, rewards = play_to_train(
                    dim, 
                    net, 
                    #exploration_prob=0.2
                    #exploration_prob=(1- epoch/epochs)
                    exploration_prob=(max(0.1, 1 - epoch/epochs))
                    )
            ep_states += states
            ep_actions += actions
            ep_rewards += rewards

        transitions = get_transitions(ep_states, ep_actions, ep_rewards)
        random.shuffle(transitions)

        true_q = []
        gamma_decay = 0.9

        for transition in transitions:
            #exp_rewards.append(calc_exp_reward(net, transition))
            true_q.append(calc_true_exp_reward(net, transition, gamma_decay))


        print("TRAIN")
        world.plot_q_table(net, filename="q_{}.png".format(network_dir + str(epoch)))
        net.train(transitions, true_q)
        world.plot_q_table(net, filename="q_{}.png".format(network_dir + str(epoch + 1)))

#    while world.snake.alive:
#        world_map = world.get_map()
#        print("active world:")
#        pprint.pprint(world_map)
#        next_action = world.get_next_action(net, exploration_prob=0)
#
#        world.snake.set_direction(next_action)
#        world.snake.move()
#        world.update_snake()
#        time.sleep(2)
#
main()
