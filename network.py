import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import glob

class Network():
    def __init__(self, chkpt="training"):  
        # input = map / state of the world
        # output = expected reward for each possible action
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(25,)),
            keras.layers.Dense(25, activation=tf.nn.relu),
            keras.layers.Dense(25, activation=tf.nn.relu),
            keras.layers.Dense(4, activation=None) #tf.nn.softmax
        ])

        self.model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

        # check point saving
        checkpoint_path = chkpt + os.path.sep + "cp.ckpt"
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

    def get_action(self, state):
        # using state, predict snake action rewards and pick the expected optimal action
        print(state)
        expected_reward = self.predict(state)
        print(expected_reward)
        return np.argmax(expected_reward) -1

    def predict(self, state): # if only used for get_action(), directly code above and rm this one
        return self.model.predict(state)


    def train(self, states, rewards, epochs=1):
        self.model.fit(
                states,
                rewards,
                epochs=epochs,
                callbacks=[self.cp_callback],
                )

    def eval(self, states, rewards):
        test_loss, test_acc = self.model.evaluate(states, rewards)
        print('Test accuracy:', test_acc)

