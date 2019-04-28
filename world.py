import sys
import numpy as np
import snake
import network
import visworld
import random

class World():
    def __init__(self, dim, this_snake=None, foods=None, obstacles=None, render=False):
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

        if render:
            self.vis = visworld.Vis(self.dim, self.BLOCKSIZE)


    def get_map(self):
        world_map = np.full(self.dim, self.EMPTY)
        for coord, segment in zip(self.snake.coords, self.snake.segments):
            world_map[coord] = segment
        for food in self.foods:
            world_map[food] = self.FOOD
        for obst in self.obstacles:
            world_map[obst] = self.OBSTACLE
        return world_map

    def draw(self):
        

    def get_state(self):
        return np.array([ord(x) for x in self.get_map().flatten()], ndmin=2)

    def get_future_state(self, action):
        pass

    def get_next_action(self, network):
        network.get_action(self.get_state())

    # mv to Snake --> needs to know whether to grow on move or die, keep reward as snake.reward
    def score(self, world_map, coord):
        x,y = coord
        if x < 0 or y < 0 or x >= world_map.shape[0] or y >= world_map.shape[1]:
            return self.OBSTACLE_SCORE # and snake.die()
        field_to_score = world_map[coord]
        if field_to_score == self.FOOD:
            return self.FOOD_SCORE
        if field_to_score == self.EMPTY:
            return self.EMPTY_SCORE
        return self.OBSTACLE_SCORE # and snake.die()
        # extend: penalize not empty field but change of direction, to favor bigger movements / laziness?  --> input current state AND previous action into network

        
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
        for action in world.snake.action_space:
            world.snake.set_direction(action)
            world.snake.move()
            rewards.append(world.score(world_map, world.snake.coords[0])) # fix snake mv for list of coords
            world.snake.coords = [(i,j)]
        simu_states.append(state)
        simu_rewards.append(rewards)
    return simu_states, simu_rewards


def main():
    world = World((5,5))
    training_dir = '01'
    net = network.Network(training_dir)

    simu_states, simu_rewards = simu_for_training(world.dim, 5)
    print(world.get_next_action(net))
    net.train(simu_states, simu_rewards)


main()
    


