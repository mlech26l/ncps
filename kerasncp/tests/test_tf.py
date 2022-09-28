# Copyright 2022 Mathias Lechner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import time
import pytest
import numpy as np
import tensorflow as tf
from kerasncp.tf.experimental import CfC


def test_fit():
    N = 48  # Length of the time-series
    # Input feature is a sine and a cosine wave
    data_x = np.stack(
        [np.sin(np.linspace(0, 3 * np.pi, N)), np.cos(np.linspace(0, 3 * np.pi, N))],
        axis=1,
    )
    data_x = np.expand_dims(data_x, axis=0).astype(np.float32)  # Add batch dimension
    # Target output is a sine with double the frequency of the input signal
    data_y = np.sin(np.linspace(0, 6 * np.pi, N)).reshape([1, N, 1]).astype(np.float32)
    print("data_y.shape: ", str(data_y.shape))
    rnn = CfC(8, return_sequences=True)
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(None, 2)),
            rnn,
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss="mean_squared_error")
    model.fit(x=data_x, y=data_y, batch_size=1, epochs=3)


def test_mm_rnn():
    N = 48  # Length of the time-series
    # Input feature is a sine and a cosine wave
    data_x = np.stack(
        [np.sin(np.linspace(0, 3 * np.pi, N)), np.cos(np.linspace(0, 3 * np.pi, N))],
        axis=1,
    )
    data_x = np.expand_dims(data_x, axis=0).astype(np.float32)  # Add batch dimension
    # Target output is a sine with double the frequency of the input signal
    data_y = np.sin(np.linspace(0, 6 * np.pi, N)).reshape([1, N, 1]).astype(np.float32)
    print("data_y.shape: ", str(data_y.shape))
    rnn = CfC(8, return_sequences=True, mixed_memory=True)
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(None, 2)),
            rnn,
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss="mean_squared_error")
    model.fit(x=data_x, y=data_y, batch_size=1, epochs=3)