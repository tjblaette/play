import numpy as np

class Snake:
    def __init__(self, pos, direction=None):
        self.pos = [pos]
        self.alive = True
        self.ACTION_SPACE_LIT = ['right', 'down', 'left', 'up']
        self.ACTION_SPACE = np.arange(len(self.ACTION_SPACE_LIT))
        self.ACTION_DIM = len(self.ACTION_SPACE)

        self.direction = direction
        if not self.direction:
            self.direction = self.ACTION_SPACE[0] # set to None, pause game on init until first direction is given
        self.last_reward = None
        self.total_reward = 0

        self.HEAD = ':'
        self.BODY = 'o'
        self.TAIL = 's'
        self.segments = self.get_segments() # str representation of snake
        self.grow_on_next_move = False

    def get_segments(self):
        inner_segments = self.BODY * (len(self.pos) -2)
        return self.HEAD + inner_segments + self.TAIL

    def set_direction(self, new_direction):
        self.direction = new_direction

    def reward(self, score):
        self.last_reward = score
        self.total_reward += score

    def grow(self):
        # grow
        self.grow_on_next_move = False
        pass

    def move(self):
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


    def die(self):
        print("You died at length {}!".format(len(self.pos)))
        self.alive = False

