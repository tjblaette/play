import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import glob

class Network():
    def __init__(self, world):  
        # network structure
        # input = state from world
        # output = action for snake (direction to move in)
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=world.snake.state.shape),
            keras.layers.Dense(5, activation=tf.nn.relu),
            keras.layers.Dense(4, activation=tf.nn.softmax)
        ])

        self.model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

        # check point saving
        checkpoint_path = "training_01/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                         save_weights_only=True,
                                                         verbose=1)

        # if a checkpoint file already exists, load it
        if glob.glob(checkpoint_path + "*"):
            self.model.load_weights(checkpoint_path)
        else:
            print("No checkpoint loaded!")
            pass

    def get_direction(self, world):
        # using state, predict action for snake
        prediction = self.predict(world)
        directions = ['right', 'down', 'left', 'up']
        return directions[prediction]

    def predict(self, world):
        #print('predict')
        #print(world.snake.state)
        #print(world.snake.state.shape)
        #print(np.expand_dims(world.snake.state, 0))
        #print(np.expand_dims(world.snake.state, 0).shape)
        #print('actual prediction:')
        #print(self.model.predict(np.expand_dims(world.snake.state, 0)))
        return np.argmax(self.model.predict(np.expand_dims(world.snake.state, 0))) -1

    # fix this, so I can actually learn on an entire batch
    def train(self, states, actions, epochs=2):
        directions = ['right', 'down', 'left', 'up']
        for _ in range(epochs):
            for state, action in zip(states, actions):
                self.model.fit(
                        np.expand_dims(state, 0), 
                        np.expand_dims(directions.index(action), 0), 
                        epochs=1,
                        callbacks=[self.cp_callback],
                        )

    def eval(self, states, actions):
        test_loss, test_acc = self.model.evaluate(states, actions)
        print('Test accuracy:', test_acc)

