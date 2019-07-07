import pygame
import sys
import numpy as np

class Vis():
    def __init__(self, render, dim):
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

        window_dim = (640, 640)
        self.BLOCKSIZE = min(
            int(window_dim[0] / dim[0]),
            int(window_dim[1] / dim[1]))
        window_dim = (dim[1] * self.BLOCKSIZE, dim[0] * self.BLOCKSIZE)

        pygame.init()
        pygame.event.set_allowed(
            [pygame.KEYDOWN,
             pygame.QUIT])
        pygame.event.set_blocked(
            [pygame.MOUSEMOTION,
             pygame.MOUSEBUTTONDOWN,
             pygame.MOUSEBUTTONUP,
             pygame.KEYUP])

        self.render = render
        if self.render:
            self.FPS = 4 # frame per second
            self.clock = pygame.time.Clock()

            self.surface = pygame.display.set_mode(window_dim)
            self.surface.fill(self.COLOR_BACKGROUND)
            pygame.display.set_caption("Let's play snake!")
        else:
            self.surface = pygame.Surface(window_dim)


    def get_state(self):
        """
        Return the state of the world based on
        the pygame visualization of it. Motivation: Scale
        state image independent of world dimension to
        train a network that works for different world.dim.
        """
        state = pygame.surfarray.array3d(self.surface)
        #state = state[::self.BLOCKSIZE, ::self.BLOCKSIZE]
        state = state[::20, ::20]

        state = rgb2gray(state)
        state = state / 255.0
        #pygame.image.save(pygame.surfarray.make_surface(state), "game_world_gray.png")
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
        self.draw(world)
        if self.render:
            pygame.display.update()


    def check_for_user_input(self, world):
        """
        Check for user input:
        Quit game on window close.
        Toggle game pause on space bar.
        Set snake direction on WASD and arrow keys.

        Note: The outer for-loop should not be required but for some
        reason pygame.event.get() returns only one key Event for me,
        though it is supposed to return the entire queue. To compensate,
        I call this function manually often enough to guarantee handling
        all key presses per tick of the clock / per snake move.
        """
        for _ in range(10000):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.end()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        world.paused = not world.paused
                    elif not world.paused:
                        if (event.key == pygame.K_w
                                or event.key == pygame.K_UP):
                            world.snake.set_direction(3)
                        if (event.key == pygame.K_s
                                or event.key == pygame.K_DOWN):
                            world.snake.set_direction(1)
                        if (event.key == pygame.K_d
                                or event.key == pygame.K_RIGHT):
                            world.snake.set_direction(0)
                        if (event.key == pygame.K_a
                                or event.key == pygame.K_LEFT):
                            world.snake.set_direction(2)

    def end(self):
        """
        End the game, close the window.
        """
        pygame.quit()
        sys.exit()


def rgb2gray(rgb):
    """
    Convert 3-dimensional RGB value array to 1-dimensional
    grayscale value array. Taken from stackoverflow:
    https://stackoverflow.com/questions/12201577/
            how-can-i-convert-an-rgb-image-into-grayscale-in-python

    Args:
        rgb: Numpy array with shape [,,3] containing RGB values
            of an image.

    Returns:
        Numpy array with shape [,,1] containing grayscale values
            of rgb input array.
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
