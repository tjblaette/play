import sys
import numpy as np
import snake
import network
import visworld
import random
import time
import pprint

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
            self.snake = snake.Snake(self.random_empty_field())
        if self.foods is None:
            self.foods = [self.random_empty_field()]
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

    def get_empty_fields(self):
        empty = []
        world_map = self.get_map()
        height, width = self.dim
        for i in range(height):
            for j in range(width):
                if world_map[i,j] == self.EMPTY:
                    empty.append((i,j))
        return empty

    def random_empty_field(self):
        return random.choice(self.get_empty_fields())


    def get_state(self):
        world_map = self.get_map()
        char_to_int = np.vectorize(lambda x: ord(x))
        ascii_map = char_to_int(world_map)
        state = ascii_map.flatten()
        return state

    def get_optimal_action(self, network):
        current_state = self.get_state()
        max_expected_reward = None
        max_expected_reward_action = None
        for action in self.snake.ACTION_SPACE:
            full_state = np.concatenate((current_state, [action]))
            full_state = np.array(full_state, ndmin=2)
            expected_reward = network.get_action(full_state)
            if max_expected_reward is None or expected_reward > max_expected_reward:
                max_expected_reward = expected_reward
                max_expected_reward_action = action
        #print("max expected reward {} for action {}".format(max_expected_reward, max_expected_reward_action))
        assert max_expected_reward is not None
        return max_expected_reward_action

    def get_random_action(self):
        return random.randrange(0,len(self.snake.ACTION_SPACE))

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
        self.foods = [self.random_empty_field()]

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

def process_rewards_for_training(rewards):
    # input is reward for each consecutive state of one episode
    decay = 0.1
    expected_future_rewards = []
    for i,reward in enumerate(rewards):
        total_future_reward = 0
        for j,future_reward in enumerate(rewards[i:]):
            total_future_reward += future_reward * decay**j
        expected_future_rewards += [total_future_reward]
    #print("rewards per state per episode: {}".format(rewards))
    #print("cum_expected_future_reward per state: {}".format(expected_future_rewards))
    return expected_future_rewards


def play_to_train(dim, network_dir, exploration_prob=1, should_render=False):
    world = World(dim)
    net = network.Network(network_dir)

    states = []
    actions = []
    rewards = []
    
    while world.snake.alive and len(actions) < 20:
        world_map = world.get_map()
        if should_render:
            pprint.pprint(world_map)

        state = world.get_state()
        action = world.get_next_action(net, exploration_prob)
        world.snake.set_direction(action)
        world.snake.move()
        world.update_snake()
        reward = world.snake.last_reward

        states.append(state.tolist())
        actions.append(action)
        rewards.append(reward)
        if should_render:
            time.sleep(2)

    rewards = process_rewards_for_training(rewards)
            
    print("Collected {} states for training".format(len(actions)))
    #print(states)
    #print(actions)
    #print(rewards)
    return states, actions, rewards

def main():
    dim = (5,5)
    network_dir = '21_rl_5-5-action_one-food_no-grow_decay01_02'
    ep_states = []
    ep_actions = []
    ep_rewards = []
    ep_action_states = []

    for i in range(1000):
        print("---------")
        print(i)
        states, actions, rewards = play_to_train(dim, network_dir, exploration_prob=1)
        action_states = [state + [action] for state, action in zip(states, actions)]
        ep_states += states
        ep_actions += actions
        ep_rewards += rewards
        ep_action_states += action_states

    indices = np.arange(len(ep_action_states))
    random.shuffle(indices)
    shuffled_action_states = [ep_action_states[i] for i in indices]
    shuffled_rewards = [ep_rewards[i] for i in indices]

    print("TRAIN")
    net = network.Network(network_dir)
    net.train(np.array(shuffled_action_states), shuffled_rewards)

    world = World(dim)
    while world.snake.alive:
        world_map = world.get_map()
        print("active world:")
        pprint.pprint(world_map)
        next_action = world.get_next_action(net, exploration_prob=0)

        world.snake.set_direction(next_action)
        world.snake.move()
        world.update_snake()
        time.sleep(2)


    

def main_old():
    world = World((5,5))
    #training_dir = '15' # reward bases on closest distance to food scaled by FOOD_SCORE
    training_dir = '16_pos-food-reward-based-on-distance-closed-already' # reward bases on FOOD_SCORE - closest distance to food 
    training_dir = '17_grow'
#    training_dir = '17' # predict optimal action directly instead of reward
    net = network.Network(training_dir)
    simu_for_training = simu_rewards_for_training
    simu_for_training = simu_actions_for_training

    print("TRAIN")
    simu_states, simu_rewards = simu_for_training(world.dim, 10000)
    net.train(simu_states, simu_rewards)

    print("PREDICT multi")
    simu_states, simu_rewards = simu_for_training(world.dim, 5)
    net.get_action(simu_states)

    print("PREDICT single")
    simu_states, simu_rewards = simu_for_training(world.dim, 1)
    net.get_action(simu_states)

    while world.snake.alive:
        world_map = world.get_map()
        print("active world:")
        pprint.pprint(world_map)
        next_action = world.get_next_action(net)

        world.snake.set_direction(next_action)
        world.snake.move()
        world.update_snake()
        time.sleep(2)

main()
