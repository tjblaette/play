import numpy as np

class Snake:
    def __init__(self, pos, direction=None):
        """
        Initialize Snake.

        Args:
            pos (int, int): Coordinate tuple dedicating the snake's
                position in the world.
            direction (int): Direction the snake should move in,
                must be element of self.ACTION_SPACE.
        """
        self.pos = [pos]
        self.alive = True
        self.grow_on_next_move = False

        self.last_reward = None
        self.total_reward = 0

        # define possible actions that the snake can perform
        self.ACTION_SPACE_LIT = ['right', 'down', 'left', 'up']
        self.ACTION_SPACE = np.arange(len(self.ACTION_SPACE_LIT))
        self.ACTION_DIM = len(self.ACTION_SPACE)

        # establish direction of movement
        # --> direction = action performed
        self.direction = direction
        if not self.direction:
            self.direction = self.ACTION_SPACE[0]

        # define string representation to print to screen
        self.HEAD = ':'
        self.BODY = 'o'
        self.TAIL = 's'
        self.segments = self.get_segments()


    def get_segments(self):
        """
        Obtain a string representation of the snake.

        Returns:
            A list of string characters.
        """
        inner_segments = self.BODY * (len(self.pos) -2)
        return self.HEAD + inner_segments + self.TAIL

    def set_direction(self, new_direction):
        """
        Set the direction that the snake will move in next.

        Args:
            new_direction (int): Sampled from self.ACTION_SPACE.
        """
        self.direction = new_direction

    def reward(self, score):
        """
        Save the score of a previous action.
        """
        self.last_reward = score
        self.total_reward += score

    def move(self):
        """
        Move the snake to the next field.
        If the previous action lead it to
        food, let it grow and become longer.
        """
        # get new head
        y,x = self.pos[0]
        if self.direction == 0:  #'right'
            x += 1
        elif self.direction == 2:  #'left'
            x -= 1
        elif self.direction == 1:  #'down'
            y += 1
        elif self.direction == 3:  #'up'
            y -= 1

        # prepend new head
        self.pos = [(y,x)] + self.pos
        if not self.grow_on_next_move:
            del self.pos[-1]
        self.grow_on_next_move = False


    def die(self, verbose):
        """
        Kill the snake.

        Args:
            verbose (bool): Whether to inform the user and
                print the snake's length at the time of death.
        """
        if verbose:
            print("You died at length {}!".format(len(self.pos)))
        self.alive = False

