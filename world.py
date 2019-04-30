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
        self.empty_score = self.score_empty_field # penalize moving onto empty fields - favor closing in on food

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


    def render(self):
        pass

    def get_state(self):
        world_map = self.get_map()
        char_to_int = np.vectorize(lambda x: ord(x))
        return char_to_int(world_map)

    def get_future_state(self, action):
        pass

    def get_next_action(self, network):
        current_state = self.get_state()
        return network.get_action(np.array(current_state, ndmin=3))

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
            # self.snake.grow_on_next_move = True
        elif self.is_snake_in_obstacle():
            self.snake.reward(self.OBSTACLE_SCORE)
            self.snake.die()
        else:
            self.snake.reward(self.score_empty_field())

    def score_empty_field(self):
        return self.FOOD_SCORE - self.distance_to_closest_food()

    def distance_to_closest_food(self):
        min_dist = None
        for food in self.foods:
            dist = sum([abs(food_coord - snake_coord) for food_coord, snake_coord in zip(food, self.snake.pos[0])])
            if min_dist is None or dist < min_dist:
                min_dist = dist
        return min_dist


def simu_actions_for_training(dim, n):
    height, width = dim
    simu_states = []
    opt_actions = []
    for _ in range(n):
        i,j = random.randint(0, height -1), random.randint(0, width -1)
        world = World(dim, snake.Snake((i,j)))
        world_map = world.get_map()
        state = world.get_state()
        rewards = []
        pprint.pprint(world_map)
        for action in world.snake.ACTION_SPACE:
            print("action to simulate: {}".format(action))
            world.snake.set_direction(action)
            world.snake.move()
            world.update_snake()
            rewards.append(world.snake.last_reward)
            # reset snake for next simulated action
            world.snake.pos = [(i,j)]
        print("rewards obtained: {}".format(rewards))
        opt_action = np.zeros(len(rewards))
        opt_action[np.argmax(rewards)] = 1
        print("optimal action: {}".format(opt_action))
        simu_states.append(state)
        opt_actions.append(opt_action)
    simu_states = np.array(simu_states)
    opt_actions = np.array(opt_actions)

    return simu_states, opt_actions

def simu_rewards_for_training(dim, n):
    height, width = dim
    simu_states = []
    simu_rewards = []
    for _ in range(n):
        i,j = random.randint(0, height -1), random.randint(0, width -1)
        world = World(dim, snake.Snake((i,j)))
        world_map = world.get_map()
        state = world.get_state()
        rewards = []
        pprint.pprint(world_map)
        for action in world.snake.ACTION_SPACE:
            print("action to simulate: {}".format(action))
            world.snake.set_direction(action)
            world.snake.move()
            world.update_snake()
            rewards.append(world.snake.last_reward)
            # reset snake for next simulated action
            world.snake.pos = [(i,j)]
        print("rewards obtained: {}".format(rewards))
        simu_states.append(state)
        simu_rewards.append(rewards)
    simu_states = np.array(simu_states)
    simu_rewards = np.array(simu_rewards)

    return simu_states, simu_rewards


def main():
    world = World((5,5))
    #training_dir = '15' # reward bases on closest distance to food scaled by FOOD_SCORE
    training_dir = '16_pos-food-reward-based-on-distance-closed-already' # reward bases on FOOD_SCORE - closest distance to food 
#    training_dir = '17' # predict optimal action directly instead of reward
    net = network.Network(training_dir)
    #simu_for_training = simu_rewards_for_training
    #simu_for_training = simu_actions_for_training

    #print("TRAIN")
    #simu_states, simu_rewards = simu_for_training(world.dim, 10000)
    #net.train(simu_states, simu_rewards)

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


