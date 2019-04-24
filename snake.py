import pygame
import numpy as np

class Snake:
    def __init__(self, color, x, y, blocksize):
        self.color = color
        self.x = x
        self.y = y
        self.width = blocksize
        self.direction = 'right'
        self.alive = True
        self.fov = 10 # field of view
        self.state = None
        self.last_reward = None
        self.total_reward = 0

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (self.x, self.y, self.width, self.width))

    def set_direction(self, direction):
        self.direction = direction

    def move(self, world):
        if self.direction == 'right':
            self.x = self.x + self.width
        elif self.direction == 'left':
            self.x = self.x - self.width
        elif self.direction == 'down':
            self.y = self.y + self.width
        elif self.direction == 'up':
            self.y = self.y - self.width
        
        if self.moved_out_of_window(world.surface):
            self.die()
        self.set_reward(world) 

    def moved_out_of_window(self, surface):
        if self.x < 0 or self.x > surface.get_width() - self.width or self.y < 0 or self.y > surface.get_height() - self.width:
            return True
        return False

    def die(self):
        print("You died!")
        self.alive = False

    def set_reward(self, world):
        if not self.alive:
            self.last_reward = -1
        elif (self.y, self.x) in world.foods:
            self.last_reward = 10
        else:
            self.last_reward = 1
        self.total_reward += self.last_reward

    def set_state(self, full_state):
        padded_state = np.pad(full_state, self.fov, 'constant', constant_values=-1)
        normed_x = int(self.x / self.width)
        normed_y = int(self.y / self.width)
        self.state = padded_state[normed_y:(normed_y + 2*self.fov +1), normed_x:(normed_x + 2*self.fov +1)]
        #print(self.state)
        #print(self.state.size)

