import pygame
from pygame.locals import *
#import sys

class Vis():
    def __init__(self, dim, blocksize=10):
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)

        self.BLOCKSIZE = blocksize
        self.dim = dim * self.blocksize

        pygame.init()
        self.surface = pygame.display.set_mode((self.BLOCKSIZE*10, self.BLOCKSIZE*10))
        self.surface.fill(self.WHITE)
        pygame.display.set_caption("Let's play snake!")

        self.FPS = 20
        self.clock = pygame.time.Clock()


    def get_state(self):
        state = pygame.surfarray.array2d(self.surface) #.swapaxes(0,1)
        state = state[::self.BLOCKSIZE, ::self.BLOCKSIZE]
        return state

    def draw(self, world_map):
        # draw world map onto self.surface
        pass

    def update(self):
        # display surface
        pass

    def keep_rendering(self):
        # listen to pygame events
        pass

    def end(self):
        pygame.quit()
        #sys.exit()

