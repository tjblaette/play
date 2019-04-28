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

        self.snake = this_snake
        self.foods = foods
        self.obstacles = obstacles

        if not self.snake:
            self.snake = snake.Snake((random.randint(0, height-1), random.randint(0, width-1)))
        if self.foods is None:
            self.foods = []
        if self.obstacles is None:
            self.obstacles = []

        self.EMPTY = ' '
        self.FOOD = 'x'
        self.OBSTACLE = '='
        self.FOOD_SCORE = width + height
        self.OBSTACLE_SCORE = -self.FOOD_SCORE
        self.EMPTY_SCORE = -1 # penalize moving onto empty fields

        self.should_render = should_render
        if self.should_render:
            self.vis = visworld.Vis(self.dim, self.BLOCKSIZE)


    def get_map(self):
        world_map = np.full(self.dim, self.EMPTY)
        for coord, segment in zip(self.snake.pos, self.snake.segments):
            world_map[coord] = segment
        for food in self.foods:
            world_map[food] = self.FOOD
        for obst in self.obstacles:
            world_map[obst] = self.OBSTACLE
        return world_map

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
        if self.snake.pos in self.foods:
            print("Snake reached food")
            return True
        return False

    def is_snake_in_obstacle(self):
        if self.snake.pos in self.obstacles:
            print("Snake ran into obstacle")
            return True
        return False

    # adjust for > 1 snake by passing snake as arg
    def check_snake(self):
        if self.is_snake_out_of_bounds():
            self.snake.reward(self.OBSTACLE_SCORE)
            self.snake.die()
        elif self.is_snake_at_food():
            # reset food coord
            self.snake.reward(self.FOOD_SCORE)
            # self.snake.grow_on_next_move = True
        elif self.is_snake_in_obstacle():
            self.snake.reward(self.OBSTACLE_SCORE)
            self.snake.die()
        else:
            self.snake.reward(self.EMPTY_SCORE)


def simu_for_training(dim, n):
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
            print(action)
            world.snake.set_direction(action)
            world.snake.move()
            world.check_snake()
            rewards.append(world.snake.last_reward)
            # reset snake for next simulated action
            world.snake.pos = [(i,j)]
        simu_states.append(state)
        simu_rewards.append(rewards)
    simu_states = np.array(simu_states)
    simu_rewards = np.array(simu_rewards)
        
    return simu_states, simu_rewards


def main():
    world = World((5,5))
    training_dir = '08'
    net = network.Network(training_dir)

    simu_states, simu_rewards = simu_for_training(world.dim, 10000)
    print("TRAIN")
    net.train(simu_states, simu_rewards)
    print("PREDICT multi")
    net.get_action(simu_states)
    print("PREDICT single")
    print(world.get_next_action(net))

    while world.snake.alive:
        world_map = world.get_map()
        next_action = world.get_next_action(net)
        pprint.pprint(world_map)

        world.snake.set_direction(next_action)
        world.snake.move()
        world.check_snake()
        time.sleep(2)


main()


