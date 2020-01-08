import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import glob

class Network():
    def __init__(
            self,
            state_space_dim,
            action_space_dim,
            chkpt_dir="training"):
        """
        Initialize neural Network, to control Snake in World.
        Input to the model will be the map / state of the World.
        Output will be the expected reward / q-value of all of the
        Snake's possible actions in a given state.

        state_space_dim (int, int): Dimension tuple of the World to
            model, corresponds to the network's number of input units.
        action_space_dim (int): Number of actions in snake.STATE_SPACE,
            corresponds to the number of output units of the network.
        chkpt_dir (str): Name of the directory in which to save the
            model after training.
        """
        # build the model using keras functional API
        #input_states = keras.layers.Input(shape=(1024, ))
        input_states = keras.layers.Input(shape=(np.product(state_space_dim), ))
        input_actions = keras.layers.Input(shape=(action_space_dim, ))

        hidden = keras.layers.Dense(50, activation=tf.nn.relu)(input_states)
        hidden = keras.layers.Dense(50, activation=tf.nn.relu)(hidden)
        hidden = keras.layers.Dense(50, activation=tf.nn.relu)(hidden)

        output = keras.layers.Dense(
            action_space_dim,
            activation="linear")(hidden)
        masked_output = keras.layers.multiply([output, input_actions])

        self.model = keras.models.Model(
            inputs=[input_states, input_actions],
            outputs=masked_output)
        self.model.compile(
            optimizer='SGD',
            loss='mse',
            metrics=['mae'],)

        # implement check point saving
        self.checkpoint_path = chkpt_dir + os.path.sep + "model.h5"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        # if a checkpoint file already exists, load it
        if glob.glob(self.checkpoint_path + "*"):
            print("Checkpoint loaded: {}".format(self.checkpoint_path))
            self.model = tf.keras.models.load_model(self.checkpoint_path)
        else:
            print("NO CHECKPOINT FOUND")
            pass

        # make sure the model output folder exists
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)


    def predict(self, state, action, verbose=False):
        """
        For a given state and action, predict the subsequent
        reward / q-value.

        Args:
            state: A flattented 1D tuple of integers representing the
                current state of the world.
            action: A 1D numpy array with one entry per possible action.
                Rewards are predicted for those actions whose index
                is 1 -> pass all 1s to predict rewards for all
                possible actions, pass a single 1 to predict rewards
                of one specific action.
            verbose (bool): Whether to print expected rewards.

        Returns:
            Expected future reward of the requested actions: A 1D numpy
            array in the same form as the input action array.

        """
        expected_rewards = self.model.predict([state,action])
        if verbose:
            print("expected rewards: {}".format(expected_rewards))
        return expected_rewards


    def train(self, transitions, true_q, epochs=1):
        """
        Train the network, then save it to file.

        Args:
            transitions: List of (state, action, reward, next_state)
                tuples to train on.
            true_q: True expected rewards / q-values for these
                transitions.
            epochs (int): Number of times to iterate over and train on
                the given transitions.
        """
        states = [transition[0] for transition in transitions]
        actions = [transition[1] for transition in transitions]
        action_vecs = [[int(i == action) for i in range(4)]
            for action in actions]

        self.model.fit(
            [states, action_vecs],
            [true_q],
            epochs=epochs,)
        print("Saving model to {}".format(self.checkpoint_path))
        self.model.save(self.checkpoint_path)
