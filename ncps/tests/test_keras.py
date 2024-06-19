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
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Run on CPU
# os.environ["KERAS_BACKEND"] = "torch"
# os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
import pytest
import ncps
from ncps.keras import CfC, LTCCell, LTC
from ncps import wirings


def test_fc():
    data_x, data_y = prepare_test_data()
    print("data_y.shape: ", str(data_y.shape))
    fc_wiring = wirings.FullyConnected(8, 1)  # 8 units, 1 of which is a motor neuron
    ltc_cell = LTCCell(fc_wiring)

    model = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            keras.layers.RNN(ltc_cell, return_sequences=True),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    model.fit(x=data_x, y=data_y, batch_size=1, epochs=3)


def prepare_test_data():
    N = 48  # Length of the time-series
    # Input feature is a sine and a cosine wave
    data_x = np.stack(
        [np.sin(np.linspace(0, 3 * np.pi, N)), np.cos(np.linspace(0, 3 * np.pi, N))],
        axis=1,
    )
    data_x = np.expand_dims(data_x, axis=0).astype(np.float32)  # Add batch dimension
    # Target output is a sine with double the frequency of the input signal
    data_y = np.sin(np.linspace(0, 6 * np.pi, N)).reshape([1, N, 1]).astype(np.float32)
    return data_x, data_y


def test_random():
    data_x, data_y = prepare_test_data()
    arch = wirings.Random(32, 1, sparsity_level=0.5)  # 32 units, 1 motor neuron
    ltc_cell = LTCCell(arch)

    model = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            keras.layers.RNN(ltc_cell, return_sequences=True),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    model.fit(x=data_x, y=data_y, batch_size=1, epochs=3)


def test_ncp():
    data_x, data_y = prepare_test_data()
    ncp_wiring = wirings.NCP(
        inter_neurons=20,  # Number of inter neurons
        command_neurons=10,  # Number of command neurons
        motor_neurons=1,  # Number of motor neurons
        sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
        inter_fanout=5,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=6,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=4,  # How many incoming synapses has each motor neuron
    )
    ltc_cell = LTCCell(ncp_wiring)

    model = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            keras.layers.RNN(ltc_cell, return_sequences=True),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    model.fit(x=data_x, y=data_y, batch_size=1, epochs=3)


def test_fit():
    data_x, data_y = prepare_test_data()
    print("data_y.shape: ", str(data_y.shape))
    rnn = CfC(8, return_sequences=True)
    model = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            rnn,
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    model.fit(x=data_x, y=data_y, batch_size=1, epochs=3)


def test_mm_rnn():
    data_x, data_y = prepare_test_data()
    print("data_y.shape: ", str(data_y.shape))
    rnn = CfC(8, return_sequences=True, mixed_memory=True)
    model = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            rnn,
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    model.fit(x=data_x, y=data_y, batch_size=1, epochs=3)


def test_ncp_rnn():
    data_x, data_y = prepare_test_data()
    ncp_wiring = wirings.NCP(
        inter_neurons=20,  # Number of inter neurons
        command_neurons=10,  # Number of command neurons
        motor_neurons=1,  # Number of motor neurons
        sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
        inter_fanout=5,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=6,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=4,  # How many incoming synapses has each motor neuron
    )
    ltc = LTC(ncp_wiring, return_sequences=True)

    model = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            ltc,
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    model.fit(x=data_x, y=data_y, batch_size=1, epochs=3)


def test_auto_ncp_rnn():
    data_x, data_y = prepare_test_data()
    ncp_wiring = wirings.AutoNCP(28, 1)
    ltc = LTC(ncp_wiring, return_sequences=True)

    model = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            ltc,
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    model.fit(x=data_x, y=data_y, batch_size=1, epochs=3)
    assert ncp_wiring.synapse_count > 0
    assert ncp_wiring.sensory_synapse_count > 0


def test_random_cfc():
    data_x, data_y = prepare_test_data()
    arch = wirings.Random(32, 1, sparsity_level=0.5)  # 32 units, 1 motor neuron
    cfc = CfC(arch, return_sequences=True)

    model = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            cfc,
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    model.fit(x=data_x, y=data_y, batch_size=1, epochs=3)


def test_ncp_cfc_rnn():
    data_x, data_y = prepare_test_data()
    ncp_wiring = wirings.NCP(
        inter_neurons=20,  # Number of inter neurons
        command_neurons=10,  # Number of command neurons
        motor_neurons=1,  # Number of motor neurons
        sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
        inter_fanout=5,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=6,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=4,  # How many incoming synapses has each motor neuron
    )
    ltc = CfC(ncp_wiring, return_sequences=True)

    model = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            ltc,
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    model.fit(x=data_x, y=data_y, batch_size=1, epochs=3)


def test_auto_ncp_cfc_rnn():
    data_x, data_y = prepare_test_data()
    ncp_wiring = wirings.AutoNCP(32, 1)
    ltc = CfC(ncp_wiring, return_sequences=True)

    model = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            ltc,
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    model.fit(x=data_x, y=data_y, batch_size=1, epochs=3)


def test_ltc_rnn():
    data_x, data_y = prepare_test_data()
    ltc = LTC(32, return_sequences=True)

    model = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            ltc,
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    model.fit(x=data_x, y=data_y, batch_size=1, epochs=3)


def test_ncps():
    input_size = 8

    wiring = ncps.wirings.FullyConnected(8, 4)  # 16 units, 8 motor neurons
    ltc_cell = LTCCell(wiring)
    data = keras.random.normal([3, input_size])
    hx = keras.ops.zeros([3, wiring.units])
    output, hx = ltc_cell(data, hx)
    assert output.shape == (3, 4)
    assert hx[0].shape == (3, wiring.units)


def test_ncp_sizes():
    wiring = ncps.wirings.NCP(10, 10, 8, 6, 6, 4, 6)
    rnn = LTC(wiring)
    data = keras.random.normal([5, 3, 8])
    output = rnn(data)
    assert wiring.synapse_count > 0
    assert wiring.sensory_synapse_count > 0
    assert output.shape == (5, 8)


def test_auto_ncp():
    wiring = ncps.wirings.AutoNCP(16, 4)
    rnn = LTC(wiring)
    data = keras.random.normal([5, 3, 8])
    output = rnn(data)
    assert output.shape == (5, 4)


def test_ncp_cfc():
    wiring = ncps.wirings.NCP(10, 10, 8, 6, 6, 4, 6)
    rnn = CfC(wiring)
    data = keras.random.normal([5, 3, 8])
    output = rnn(data)
    assert output.shape == (5, 8)


def test_auto_ncp_cfc():
    wiring = ncps.wirings.AutoNCP(28, 10)
    rnn = CfC(wiring)
    data = keras.random.normal([5, 3, 8])
    output = rnn(data)
    assert output.shape == (5, 10)
