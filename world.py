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
        for action in world.snake.ACTION_SPACE:
            world.snake.set_direction(action)
            world.snake.move()
            rewards.append(world.score(world_map, world.snake.pos[0])) # fix snake mv for list of pos
            world.snake.pos = [(i,j)]
        simu_states.append(state)
        simu_rewards.append(rewards)
    simu_states = np.array(simu_states)
    simu_rewards = np.array(simu_rewards)
        
    return simu_states, simu_rewards


def main():
    world = World((5,5))
    training_dir = '03'
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
        if world.score(world_map, world.snake.pos[0]) == world.OBSTACLE_SCORE: # check map at current snake pos before it moved there (== did it die?)  --> implement better is_dead() fct
            world.snake.die()
        time.sleep(2)
    


main()
    


