# Copyright (2017-2020)
# The Wormnet project
# Mathias Lechner (mlechner@ist.ac.at)
import numpy as np
import tensorflow as tf
import kerasncp as kncp
from kerasncp.tf import LTCCell


data_x = np.random.default_rng().normal(size=(100, 16, 10))
data_y = np.random.default_rng().normal(size=(100, 16, 1))
print("data_y.shape: ", str(data_y.shape))

wiring = kncp.wirings.FullyConnected(16, 8)  # 16 units, 8 motor neurons
rnn_cell = LTCCell(wiring)

dense1 = tf.keras.layers.Dense(16, activation="tanh")
dense2 = tf.keras.layers.Dense(1)

inputs = tf.keras.Input(shape=(None, 10))
x = dense1(inputs)
x = tf.keras.layers.RNN(rnn_cell, return_sequences=True)(x)
x = dense2(x)
trainable_model = tf.keras.Model(inputs, x)
trainable_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.MeanSquaredError()
)
trainable_model.fit(x=data_x, y=data_y, batch_size=25, epochs=10)
trainable_model.evaluate(x=data_x, y=data_y)

# Now we need to construct a single-step model that accepts an initial hidden state as additional input
inputs_single = tf.keras.Input(shape=(10,))
inputs_state = tf.keras.Input(shape=(rnn_cell.state_size,))
x = dense1(inputs_single)
_, output_states = rnn_cell(x, inputs_state)
single_step_model = tf.keras.Model([inputs_single, inputs_state], output_states)


def infer_hidden_states(single_step_model, state_size, data_x):
    """
        Infers the hidden states of a single-step RNN model
    Args:
        single_step_model: RNN model taking a pair (inputs,old_hidden_state) as input and outputting new_hidden_state
        state_size: Size of the RNN model (=number of units)
        data_x: Input data for which the hidden states should be inferred

    Returns:
        Tensor of shape (batch_size,sequence_length+1,state_size). The sequence starts with the initial hidden state
        (all zeros) and is therefore one time-step longer than the input sequence
    """
    batch_size = data_x.shape[0]
    seq_len = data_x.shape[1]
    hidden = tf.zeros((batch_size, state_size))
    hidden_states = [hidden]
    for t in range(seq_len):
        # Compute new hidden state from old hidden state + input at time t
        print("hidden.shape", hidden)
        hidden = single_step_model([data_x[:, t], hidden])
        print("all", hidden)
        print("all", len(hidden))
        hidden_states.append(hidden)
    return tf.stack(hidden_states, axis=1)


# Now we can infer the hidden state
states = infer_hidden_states(single_step_model, rnn_cell.state_size, data_x)
print("Hidden states of first example ", states[0])

for i in range(wiring.units):
    print("Neuron {:0d} is a {:} neuron".format(i, wiring.get_type_of_neuron(i)))