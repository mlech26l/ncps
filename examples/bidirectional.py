# Copyright (2017-2020)
# The Wormnet project
# Mathias Lechner (mlechner@ist.ac.at)
import numpy as np
import tensorflow as tf
from kerasncp import wirings
from kerasncp.tf import LTCCell

N = 48  # Length of the time-series
# Input feature is a sine and a cosine wave
data_x = np.stack(
    [np.sin(np.linspace(0, 3 * np.pi, N)), np.cos(np.linspace(0, 3 * np.pi, N))], axis=1
)
data_x = np.expand_dims(data_x, axis=0).astype(np.float32)  # Add batch dimension
# Target output is a sine with double the frequency of the input signal
data_y = np.sin(np.linspace(0, 6 * np.pi, N)).reshape([1, N, 1]).astype(np.float32)
print("data_y.shape: ", str(data_y.shape))

fc_wiring = wirings.FullyConnected(8, 1)  # 8 units, 1 of which is a motor neuron
ltc_cell = LTCCell(fc_wiring)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(None, 2)),
        tf.keras.layers.Bidirectional(tf.keras.layers.RNN(ltc_cell)),
        tf.keras.layers.Dense(1)
        # tf.keras.layers.RNN(ltc_cell, return_sequences=True),
    ]
)
model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss="mean_squared_error")

model.summary()
model.fit(x=data_x, y=data_y, batch_size=1, epochs=200)
