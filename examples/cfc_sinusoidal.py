# Copyright (2017-2021)
# The Wormnet project
# Mathias Lechner (mlechner@ist.ac.at)
# Do not distribute without permission
import numpy as np
import os
import tensorflow as tf
import kerasncp as kncp
from kerasncp.tf import WiredCfcCell, MixedMemoryRNN

N = 48
data_x = np.stack(
    [np.sin(np.linspace(0, 3 * np.pi, N)), np.cos(np.linspace(0, 3 * np.pi, N))], axis=1
)
data_x = np.expand_dims(data_x, axis=0).astype(np.float32)  # Add batch dimension
data_y = np.sin(np.linspace(0, 6 * np.pi, N)).reshape([1, N, 1]).astype(np.float32)
print("data_x.shape: ", str(data_x.shape))
print("data_y.shape: ", str(data_y.shape))

# arch = kncp.FullyConnected(8, 1)  # 8 units, 1 motor neuron
arch = kncp.wirings.NCP(
    inter_neurons=16,
    command_neurons=8,
    motor_neurons=1,
    sensory_fanout=12,
    inter_fanout=4,
    recurrent_command_synapses=5,
    motor_fanin=8,
)  # 8 units, 1 motor neuron
# arch = kncp.wirings.Random(8, 1, sparsity_level=0.5)  # 8 units, 1 motor neuron
rnn_cell = WiredCfcCell(arch, mode="default")
arch2 = kncp.wirings.Random(8, 1, sparsity_level=0.5)  # 8 units, 1 motor neuron
rnn_cell2 = WiredCfcCell(arch2, mode="pure")
rnn_cell3 = WiredCfcCell(arch2, mode="no_gate")
mm_cell = MixedMemoryRNN(rnn_cell3)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(None, 2)),
        tf.keras.layers.RNN(rnn_cell, return_sequences=True),
    ]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.MeanSquaredError()
)

model.summary()

model.fit(x=data_x, y=data_y, batch_size=1, epochs=400)