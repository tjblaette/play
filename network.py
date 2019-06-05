import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import glob

class Network():
    def __init__(self, state_space_dim, action_space_dim, chkpt_dir="training"):
        # input = map / state of the world
        # output = expected reward for each possible action
        input_states = keras.layers.Input(shape=(state_space_dim, ))
        input_actions = keras.layers.Input(shape=(action_space_dim, ))
        
        hidden = keras.layers.Dense(50, activation=tf.nn.relu)(input_states)
        hidden = keras.layers.Dense(50, activation=tf.nn.relu)(hidden)

        output = keras.layers.Dense(action_space_dim, activation="linear")(hidden) 
        masked_output = keras.layers.multiply([output, input_actions])

        self.model = keras.models.Model(inputs=[input_states, input_actions], outputs=masked_output)
        self.model.compile(optimizer='adam',
                loss='mse',
                metrics=['mae'],
                )

        # check point saving
        self.checkpoint_path = chkpt_dir + os.path.sep + "model.h5"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        # if a checkpoint file already exists, load it
        if glob.glob(self.checkpoint_path + "*"):
            print("Checkpoint loaded: {}".format(self.checkpoint_path))
            self.model = tf.keras.models.load_model(self.checkpoint_path)
        else:
            print("NO CHECKPOINT FOUND")
            pass

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)


    def predict(self, state, action):
        # for a given state and action, predict subsequent reward
        expected_rewards = self.model.predict([state,action])
        print("expected rewards: {}".format(expected_rewards))
        return expected_rewards

    def train(self, transitions, true_q, epochs=1):
        states = [transition[0] for transition in transitions]
        actions = [transition[1] for transition in transitions]
        action_vecs = [[int(i == action) for i in range(4)] for action in actions]

        self.model.fit(
                [states, action_vecs],
                [true_q],
                epochs=epochs,
                )
        print("Saving model to {}".format(self.checkpoint_path))
        self.model.save(self.checkpoint_path)
