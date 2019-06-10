import pygame
import sys
from pygame.locals import *

class Vis():
    def __init__(self, dim):
        self.RED = (214, 68, 23)
        self.GREEN = (33, 206, 151)
        self.BLUE = (66, 134, 244)
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.LIGHT_GRAY = (239, 239, 239)
        self.DARK_GRAY = (20, 20, 20)

        self.COLOR_SNAKE = self.BLUE
        self.COLOR_FOOD = self.GREEN
        self.COLOR_OBSTACLE = self.BLACK
        self.COLOR_BACKGROUND = self.DARK_GRAY

        self.BLOCKSIZE = 200
        self.dim = (dim[0] * self.BLOCKSIZE, dim[1] * self.BLOCKSIZE)

        pygame.init()
        self.surface = pygame.display.set_mode(self.dim)
        self.surface.fill(self.COLOR_BACKGROUND)
        pygame.display.set_caption("Let's play snake!")

        self.FPS = 2
        self.clock = pygame.time.Clock()


    def get_state(self):
        state = pygame.surfarray.array2d(self.surface) #.swapaxes(0,1)
        state = state[::self.BLOCKSIZE, ::self.BLOCKSIZE]
        return state

    def draw(self, world):
        self.surface.fill(self.COLOR_BACKGROUND)
        # draw snake
        # distinguish head once snake can grow!
        for segment in world.snake.pos:
            pygame.draw.rect(
                    self.surface,
                    self.COLOR_SNAKE,
                    (segment[1]*self.BLOCKSIZE, segment[0]*self.BLOCKSIZE, self.BLOCKSIZE, self.BLOCKSIZE)
                    )

        # draw food
        for food in world.foods:
            pygame.draw.circle(
                    self.surface,
                    self.COLOR_FOOD,
                    (int(food[1]*self.BLOCKSIZE+0.5*self.BLOCKSIZE), int(food[0]*self.BLOCKSIZE+0.5*self.BLOCKSIZE)),
                    int(self.BLOCKSIZE * 0.5)
                    )

        # draw obstacles
        for obstacle in world.obstacles:
            pygame.draw.rect(
                    self.surface,
                    self.COLOR_OBSTACLE,
                    (obstacle[1]*self.BLOCKSIZE, obstacle[0]*self.BLOCKSIZE, self.BLOCKSIZE, self.BLOCKSIZE)
                    )

    def update(self, world):
        self.check_for_window_close()
        self.draw(world)
        pygame.display.update()
        self.clock.tick(self.FPS)

    def check_for_window_close(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.end()

    def keep_rendering(self):
        # listen to pygame events
        pass

    def end(self):
        pygame.quit()
        sys.exit()

