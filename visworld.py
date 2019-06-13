import pygame
import sys

class Vis():
    def __init__(self, dim):
        """
        World visualization using pygame.

        dim (int, int): dimensions of the world to visualize.
        """
        RED = (214, 68, 23)
        GREEN = (33, 206, 151)
        BLUE = (66, 134, 244)
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        LIGHT_GRAY = (239, 239, 239)
        DARK_GRAY = (20, 20, 20)

        self.COLOR_SNAKE = BLUE
        self.COLOR_FOOD = GREEN
        self.COLOR_OBSTACLE = BLACK
        self.COLOR_BACKGROUND = DARK_GRAY

        window_dim = (1000, 1000)
        self.BLOCKSIZE = min(
            int(window_dim[0] / dim[0]),
            int(window_dim[1] / dim[1]))
        window_dim = (dim[0] * self.BLOCKSIZE, dim[1] * self.BLOCKSIZE)

        pygame.init()
        self.FPS = 4 # frame per second
        self.clock = pygame.time.Clock()

        self.surface = pygame.display.set_mode(window_dim)
        self.surface.fill(self.COLOR_BACKGROUND)
        pygame.display.set_caption("Let's play snake!")


    def get_state(self):
        """
            ---  NOT IMPLEMENTED YET!  ----
        TODO:
        Return the state of the world based on
        the pygame visualization of it. Motivation: Scale
        state image independent of world dimension to
        train a network that works for different world.dim.
        """
        state = pygame.surfarray.array2d(self.surface)
        state = state[::self.BLOCKSIZE, ::self.BLOCKSIZE]
        return state

    def draw(self, world):
        """
        Create a window surface and draw / fill in  all
        snake, food and obstacle tiles.
        """
        self.surface.fill(self.COLOR_BACKGROUND)
        # draw snake
        # TODO: distinguish head once snake can grow!
        for segment in world.snake.pos:
            pygame.draw.rect(
                self.surface,
                self.COLOR_SNAKE,
                (
                    segment[1]*self.BLOCKSIZE,
                    segment[0]*self.BLOCKSIZE,
                    self.BLOCKSIZE,
                    self.BLOCKSIZE))

        # draw food
        for food in world.foods:
            pygame.draw.circle(
                self.surface,
                self.COLOR_FOOD,
                (
                    int(food[1]*self.BLOCKSIZE+0.5*self.BLOCKSIZE),
                    int(food[0]*self.BLOCKSIZE+0.5*self.BLOCKSIZE)),
                int(self.BLOCKSIZE * 0.5))

        # draw obstacles
        for obstacle in world.obstacles:
            pygame.draw.rect(
                self.surface,
                self.COLOR_OBSTACLE,
                (
                    obstacle[1]*self.BLOCKSIZE,
                    obstacle[0]*self.BLOCKSIZE,
                    self.BLOCKSIZE,
                    self.BLOCKSIZE))

    def update(self, world):
        """
        Show the current state of the world.
        """
        self.check_for_window_close()
        self.draw(world)
        pygame.display.update()
        self.clock.tick(self.FPS)

    def check_for_window_close(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.end()

    def keep_rendering(self):
        # listen to pygame events
        pass

    def end(self):
        """
        End the game, close the window.
        """
        pygame.quit()
        sys.exit()

