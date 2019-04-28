import numpy as np

class Snake:
    def __init__(self, pos, direction=None):
        self.pos = [pos]
        self.alive = True
        self.ACTION_SPACE_LIT = ['right', 'down', 'left', 'up']
        self.ACTION_SPACE = np.arange(len(self.ACTION_SPACE_LIT))

        self.direction = direction
        if not self.direction:
            self.direction = self.ACTION_SPACE[0] # set to None, pause game on init until first direction is given
        self.previous_action = None

        self.HEAD = ':'
        self.BODY = 'o'
        self.TAIL = 's'
        self.segments = self.get_segments() # str representation of snake

    def get_segments(self):
        inner_segments = self.BODY * (len(self.pos) -2)
        return self.HEAD + inner_segments + self.TAIL

    def set_direction(self, new_direction):
        self.direction = new_direction


    def move(self):
        y,x = self.pos[0]
        if self.direction == 0:  #'right'
            x += 1
        elif self.direction == 2:  #'left'
            x -= 1
        elif self.direction == 1:  #'down'
            y += 1
        elif self.direction == 3:  #'up'
            y -= 1
        self.pos[0] = (y,x)
        # fix for len > 1!
        self.previous_action = self.direction # save direction that got snake to where it is now

        
    def die(self):
        print("You died!")
        self.alive = False

