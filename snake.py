
class Snake:
    def __init__(self, pos, direction=None):
        self.coords = [pos]
        self.alive = True
        self.action_space = ['right', 'down', 'left', 'up']
        self.direction = direction
        if not self.direction:
            self.direction = self.action_space[0] # set to None, pause game on init until first direction is given
        self.previous_action = None

        self.HEAD = ':'
        self.BODY = 'o'
        self.TAIL = 's'
        self.segments = self.get_segments() # str representation of snake

    def get_segments(self):
        inner_segments = self.BODY * (len(self.coords) -2)
        return self.HEAD + inner_segments + self.TAIL

    def set_direction(self, new_direction):
        self.direction = new_direction


    def move(self):
        y,x = self.coords[0]
        if self.direction == 'right':
            x += 1
        elif self.direction == 'left':
            x -= 1
        elif self.direction == 'down':
            y += 1
        elif self.direction == 'up':
            y -= 1
        self.coords[0] = (y,x)
        # fix for len > 1!
        self.previous_action = self.direction

        
    def die(self):
        print("You died!")
        self.alive = False

