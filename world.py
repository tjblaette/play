import pygame
from pygame.locals import *
import sys
import numpy as np
import snake
import network
import random

class World():
    def __init__(self, blocksize=10):
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)

        pygame.init()
        self.BLOCKSIZE = blocksize
        self.surface = pygame.display.set_mode((self.BLOCKSIZE*10, self.BLOCKSIZE*10))
        pygame.display.set_caption("Let's play snake!")
        self.surface.fill(self.WHITE)

        self.FPS = 20
        self.clock = pygame.time.Clock()

        self.snake = snake.Snake(self.BLUE, random.randint(0,self.surface.get_height() - self.BLOCKSIZE), random.randint(0,self.surface.get_width() - self.BLOCKSIZE), self.BLOCKSIZE) 
        self.snake.set_state(self.get_state())
        self.network = network.Network(self)
        self.obstacles = []
        self.foods = []


    def get_state(self):
        state = pygame.surfarray.array2d(self.surface).swapaxes(0,1)
        state = state[::self.BLOCKSIZE, ::self.BLOCKSIZE]
        return state

    def update(self):
        if self.snake.alive:
            self.snake.draw(self.surface)
            full_state = self.get_state()
            pygame.display.update()
            self.snake.set_state(full_state)
            self.clock.tick(self.FPS)

    def move(self):
        if self.snake.alive:
            self.snake.move(self)

def game_over(states, actions, last_rewards, total_rewards):
    pygame.quit()
    print("non-lethal states: {}".format(len(states)))
    return states, actions, last_rewards, total_rewards


def play():
    world = World()
    world.update()
    paused = False

    states = []
    actions = []
    last_rewards = []
    total_rewards = []

    while True:
        world.surface.fill(world.WHITE)
        for event in pygame.event.get():
            if event.type == QUIT:
                return game_over(states, actions, last_rewards, total_rewards)
            if event.type == KEYUP:
                if event.key == K_SPACE:
                    paused = False
                if event.key == K_w or event.key == K_UP:
                    world.snake.set_direction('up')
                if event.key == K_s or event.key == K_DOWN:
                    world.snake.set_direction('down')
                if event.key == K_d or event.key == K_RIGHT:
                    world.snake.set_direction('right')
                if event.key == K_a or event.key == K_LEFT:
                    world.snake.set_direction('left')
        if not paused and world.snake.alive:
            #print('get prediction from network')
            #print(world.network.get_direction(world))
            states.append(world.snake.state)
            world.snake.set_direction(world.network.get_direction(world))
            actions.append(world.snake.direction)
            world.move()
            last_rewards.append(world.snake.last_reward)
            total_rewards.append(world.snake.total_reward)
            world.update()
        elif not world.snake.alive:
            return game_over(states, actions, last_rewards, total_rewards)


def main():
    states = []
    actions = []
    last_rewards = []
    total_rewards = []
    should_train = True

    for _ in range(30):
        print(_)
        new_states, new_actions, new_last_rewards, new_total_rewards = play()

        # the final action was lethal 
        # --> for training, propose a different action (which is hopefully non-lethal)
        action_space = ['right', 'down', 'left', 'up']
        new_actions[-1] = [action for action in action_space if action != new_actions[-1]][random.randint(0,2)]

        states = states + new_states
        actions = actions + new_actions
        last_rewards = last_rewards + new_last_rewards
        total_rewards = total_rewards + new_total_rewards

        # train on those (sequences of) states that achieved a high reward
        # --> hopefully, this will improve model over time?
        median_total_reward = np.median(total_rewards)
        print("median total reward: {}".format(median_total_reward))
        states_for_training = [state for state, reward in zip(states, total_rewards) if reward <= median_total_reward]
        actions_for_training = [action for action, reward in zip(actions, total_rewards) if reward <= median_total_reward]

    if should_train:
        world = World()
        world.network.train(states_for_training, actions_for_training)

    pygame.quit()
    sys.exit()

main()
