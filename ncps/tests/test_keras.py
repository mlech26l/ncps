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


def test_bidirectional_ltc():
    rnn = keras.layers.Bidirectional(LTC(28))
    data = keras.random.normal([5, 3, 8])
    output = rnn(data)
    assert output.shape == (5, 28 * 2)


def test_bidirectional_ltc_mixed_memory():
    rnn = keras.layers.Bidirectional(LTC(28, mixed_memory=True))
    data = keras.random.normal([5, 3, 8])
    output = rnn(data)
    assert output.shape == (5, 28 * 2)


def test_bidirectional_auto_ncp_ltc():
    wiring = ncps.wirings.AutoNCP(28, 10)
    rnn = keras.layers.Bidirectional(LTC(wiring))
    data = keras.random.normal([5, 3, 8])

    output = rnn(data)
    assert output.shape == (5, 10 * 2)


def test_fit_bidirectional_auto_ncp_ltc():
    data_x, data_y = prepare_test_data()
    print("data_y.shape: ", str(data_y.shape))
    wiring = ncps.wirings.AutoNCP(28, 10)
    model = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            keras.layers.Bidirectional(LTC(wiring)),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    model.fit(x=data_x, y=data_y, batch_size=1, epochs=3)


def test_bidirectional_cfc():
    rnn = keras.layers.Bidirectional(CfC(28))
    data = keras.random.normal([5, 3, 8])
    output = rnn(data)
    assert output.shape == (5, 28 * 2)


def test_bidirectional_cfc_mixed_memory():
    rnn = keras.layers.Bidirectional(CfC(28, mixed_memory=True))
    data = keras.random.normal([5, 3, 8])
    output = rnn(data)
    assert output.shape == (5, 28 * 2)


def test_bidirectional_auto_ncp_cfc():
    wiring = ncps.wirings.AutoNCP(28, 10)
    rnn = keras.layers.Bidirectional(CfC(wiring))
    data = keras.random.normal([5, 3, 8])
    output = rnn(data)
    assert output.shape == (5, 10 * 2)


def test_bidirectional_auto_ncp_cfc_mixed_memory():
    wiring = ncps.wirings.AutoNCP(28, 10)
    rnn = keras.layers.Bidirectional(CfC(wiring, mixed_memory=True))
    data = keras.random.normal([5, 3, 8])
    output = rnn(data)
    assert output.shape == (5, 10 * 2)


def test_fit_bidirectional_auto_ncp_cfc():
    data_x, data_y = prepare_test_data()
    print("data_y.shape: ", str(data_y.shape))
    wiring = ncps.wirings.AutoNCP(28, 10)
    model = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            keras.layers.Bidirectional(CfC(wiring)),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    model.fit(x=data_x, y=data_y, batch_size=1, epochs=3)


def test_fit_bidirectional_auto_ncp_ltc_mixed_memory():
    data_x, data_y = prepare_test_data()
    print("data_y.shape: ", str(data_y.shape))
    wiring = ncps.wirings.AutoNCP(28, 10)
    model = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            keras.layers.Bidirectional(LTC(wiring, mixed_memory=True)),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    model.fit(x=data_x, y=data_y, batch_size=1, epochs=3)


def test_wiring_graph_auto_ncp_ltc():
    data_x, data_y = prepare_test_data()
    print("data_y.shape: ", str(data_y.shape))
    wiring = ncps.wirings.AutoNCP(28, 10)
    model = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            LTC(wiring),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    graph = wiring.get_graph()
    assert len(graph) == (wiring.units + 2)


def test_wiring_graph_bidirectional_auto_ncp_ltc():
    data_x, data_y = prepare_test_data()
    print("data_y.shape: ", str(data_y.shape))
    wiring = ncps.wirings.AutoNCP(28, 10)
    biLTC = keras.layers.Bidirectional(LTC(wiring))
    model = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            biLTC,
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    assert wiring.input_dim is None  # This happens because Bidirectional creates two new copies
    assert isinstance(biLTC.forward_layer, LTC)
    assert isinstance(biLTC.backward_layer, LTC)

    forward_graph = biLTC.forward_layer.cell.wiring.get_graph()
    assert len(forward_graph) == (biLTC.forward_layer.cell.wiring.units + 2)
    backward_graph = biLTC.backward_layer.cell.wiring.get_graph()
    assert len(backward_graph) == (biLTC.backward_layer.cell.wiring.units + 2)


def test_bidirectional_equivalence_ltc():
    data_x, data_y = prepare_test_data()
    print("data_y.shape: ", str(data_y.shape))
    model1 = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            keras.layers.Bidirectional(LTC(10, return_sequences=True)),
            keras.layers.Dense(1),
        ]
    )
    model2 = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            keras.layers.Bidirectional(LTC(10, return_sequences=True),
                                       backward_layer=LTC(10, return_sequences=True, go_backwards=True)),
            keras.layers.Dense(1),
        ]
    )
    model1.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    model2.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    bi_layer1 = model1.layers[0]
    bi_layer2 = model2.layers[0]
    assert isinstance(bi_layer1, keras.layers.Bidirectional)
    assert isinstance(bi_layer2, keras.layers.Bidirectional)

    fw1_config = bi_layer1.forward_layer.get_config()
    fw2_config = bi_layer2.forward_layer.get_config()
    bw1_config = bi_layer1.backward_layer.get_config()
    bw2_config = bi_layer2.backward_layer.get_config()

    def prune_details(config):
        del config['name']
        del config['cell']['config']['name']
        config['units'] = config['units'].get_config()

    prune_details(fw1_config)
    prune_details(fw2_config)
    prune_details(bw1_config)
    prune_details(bw2_config)

    assert fw1_config == fw2_config
    assert bw1_config == bw2_config

    assert isinstance(bi_layer1.forward_layer.cell.wiring, ncps.wirings.FullyConnected)
    assert isinstance(bi_layer1.backward_layer.cell.wiring, ncps.wirings.FullyConnected)
    assert isinstance(bi_layer2.forward_layer.cell.wiring, ncps.wirings.FullyConnected)
    assert isinstance(bi_layer2.backward_layer.cell.wiring, ncps.wirings.FullyConnected)


def test_bidirectional_equivalence_ltc_ncp():
    data_x, data_y = prepare_test_data()
    print("data_y.shape: ", str(data_y.shape))
    wiring = ncps.wirings.AutoNCP(28, 10)
    model1 = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            keras.layers.Bidirectional(LTC(wiring, return_sequences=True)),
            keras.layers.Dense(1),
        ]
    )
    model2 = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            keras.layers.Bidirectional(LTC(wiring, return_sequences=True),
                                       backward_layer=LTC(wiring, return_sequences=True, go_backwards=True)),
            keras.layers.Dense(1),
        ]
    )
    model1.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    model2.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    bi_layer1 = model1.layers[0]
    bi_layer2 = model2.layers[0]
    assert isinstance(bi_layer1, keras.layers.Bidirectional)
    assert isinstance(bi_layer2, keras.layers.Bidirectional)

    fw1_config = bi_layer1.forward_layer.get_config()
    fw2_config = bi_layer2.forward_layer.get_config()
    bw1_config = bi_layer1.backward_layer.get_config()
    bw2_config = bi_layer2.backward_layer.get_config()

    def prune_details(config):
        del config['name']
        del config['cell']['config']['name']
        config['units'] = config['units'].get_config()

    prune_details(fw1_config)
    prune_details(fw2_config)
    prune_details(bw1_config)
    prune_details(bw2_config)

    assert fw1_config == fw2_config
    assert bw1_config == bw2_config

    assert isinstance(bi_layer1.forward_layer.cell.wiring, ncps.wirings.AutoNCP)
    assert isinstance(bi_layer1.backward_layer.cell.wiring, ncps.wirings.AutoNCP)
    assert isinstance(bi_layer2.forward_layer.cell.wiring, ncps.wirings.AutoNCP)
    assert isinstance(bi_layer2.backward_layer.cell.wiring, ncps.wirings.AutoNCP)


def test_bidirectional_equivalence_cfc_ncp():
    data_x, data_y = prepare_test_data()
    print("data_y.shape: ", str(data_y.shape))
    wiring = ncps.wirings.AutoNCP(28, 10)
    model1 = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            keras.layers.Bidirectional(CfC(wiring, return_sequences=True)),
            keras.layers.Dense(1),
        ]
    )
    model2 = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            keras.layers.Bidirectional(CfC(wiring, return_sequences=True),
                                       backward_layer=CfC(wiring, return_sequences=True, go_backwards=True)),
            keras.layers.Dense(1),
        ]
    )
    model1.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    model2.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    bi_layer1 = model1.layers[0]
    bi_layer2 = model2.layers[0]
    assert isinstance(bi_layer1, keras.layers.Bidirectional)
    assert isinstance(bi_layer2, keras.layers.Bidirectional)

    fw1_config = bi_layer1.forward_layer.get_config()
    fw2_config = bi_layer2.forward_layer.get_config()
    bw1_config = bi_layer1.backward_layer.get_config()
    bw2_config = bi_layer2.backward_layer.get_config()

    def prune_details(config):
        del config['name']
        del config['activation']
        del config['cell']['config']['name']
        config['units'] = config['units'].get_config()
        config['wiring'] = config['wiring'].get_config()

    prune_details(fw1_config)
    prune_details(fw2_config)
    prune_details(bw1_config)
    prune_details(bw2_config)

    assert fw1_config == fw2_config
    assert bw1_config == bw2_config

    assert isinstance(bi_layer1.forward_layer.cell.wiring, ncps.wirings.AutoNCP)
    assert isinstance(bi_layer1.backward_layer.cell.wiring, ncps.wirings.AutoNCP)
    assert isinstance(bi_layer2.forward_layer.cell.wiring, ncps.wirings.AutoNCP)
    assert isinstance(bi_layer2.backward_layer.cell.wiring, ncps.wirings.AutoNCP)


def test_bidirectional_equivalence_cfc_ncp_mixed_memory():
    data_x, data_y = prepare_test_data()
    print("data_y.shape: ", str(data_y.shape))
    wiring = ncps.wirings.AutoNCP(28, 10)
    model1 = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            keras.layers.Bidirectional(CfC(wiring, return_sequences=True, mixed_memory=True)),
            keras.layers.Dense(1),
        ]
    )
    model2 = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            keras.layers.Bidirectional(CfC(wiring, return_sequences=True, mixed_memory=True),
                                       backward_layer=CfC(wiring, return_sequences=True, mixed_memory=True,
                                                          go_backwards=True)),
            keras.layers.Dense(1),
        ]
    )
    model1.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    model2.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    bi_layer1 = model1.layers[0]
    bi_layer2 = model2.layers[0]
    assert isinstance(bi_layer1, keras.layers.Bidirectional)
    assert isinstance(bi_layer2, keras.layers.Bidirectional)

    fw1_mm_cell = bi_layer1.forward_layer.cell
    fw2_mm_cell = bi_layer2.forward_layer.cell
    bw1_mm_cell = bi_layer1.backward_layer.cell
    bw2_mm_cell = bi_layer2.backward_layer.cell
    assert isinstance(fw1_mm_cell, ncps.keras.MixedMemoryRNN) and isinstance(fw2_mm_cell, ncps.keras.MixedMemoryRNN)
    assert isinstance(bw1_mm_cell, ncps.keras.MixedMemoryRNN) and isinstance(bw2_mm_cell, ncps.keras.MixedMemoryRNN)

    def prune_mm_details(config):
        del config['rnn_cell']['name']
        del config['rnn_cell']['activation']
        del config['rnn_cell']['wiring']  # Checked below
        return config

    assert prune_mm_details(fw1_mm_cell.get_config()) == prune_mm_details(fw2_mm_cell.get_config())
    assert prune_mm_details(bw1_mm_cell.get_config()) == prune_mm_details(bw2_mm_cell.get_config())

    def prune_cfc_details(config):
        del config['name']
        del config['activation']
        config['wiring'] = config['wiring'].get_config()
        return config

    assert prune_cfc_details(fw1_mm_cell.rnn_cell.get_config()) == prune_cfc_details(fw2_mm_cell.rnn_cell.get_config())
    assert prune_cfc_details(bw1_mm_cell.rnn_cell.get_config()) == prune_cfc_details(bw2_mm_cell.rnn_cell.get_config())

    assert isinstance(fw1_mm_cell.rnn_cell.wiring, ncps.wirings.AutoNCP)
    assert isinstance(fw2_mm_cell.rnn_cell.wiring, ncps.wirings.AutoNCP)
    assert isinstance(bw1_mm_cell.rnn_cell.wiring, ncps.wirings.AutoNCP)
    assert isinstance(bw2_mm_cell.rnn_cell.wiring, ncps.wirings.AutoNCP)


def test_bidirectional_equivalence_cfc():
    data_x, data_y = prepare_test_data()
    print("data_y.shape: ", str(data_y.shape))
    model1 = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            keras.layers.Bidirectional(CfC(28, return_sequences=True)),
            keras.layers.Dense(1),
        ]
    )
    model2 = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, 2)),
            keras.layers.Bidirectional(CfC(28, return_sequences=True),
                                       backward_layer=CfC(28, return_sequences=True, go_backwards=True)),
            keras.layers.Dense(1),
        ]
    )
    model1.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    model2.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    bi_layer1 = model1.layers[0]
    bi_layer2 = model2.layers[0]
    assert isinstance(bi_layer1, keras.layers.Bidirectional)
    assert isinstance(bi_layer2, keras.layers.Bidirectional)

    fw1_config = bi_layer1.forward_layer.get_config()
    fw2_config = bi_layer2.forward_layer.get_config()
    bw1_config = bi_layer1.backward_layer.get_config()
    bw2_config = bi_layer2.backward_layer.get_config()

    def prune_details(config):
        del config['name']
        del config['activation']
        del config['cell']['config']['name']

    prune_details(fw1_config)
    prune_details(fw2_config)
    prune_details(bw1_config)
    prune_details(bw2_config)

    assert fw1_config == fw2_config
    assert bw1_config == bw2_config

    assert isinstance(bi_layer1.forward_layer.cell, ncps.keras.CfCCell)
    assert isinstance(bi_layer1.backward_layer.cell, ncps.keras.CfCCell)
    assert isinstance(bi_layer2.forward_layer.cell, ncps.keras.CfCCell)
    assert isinstance(bi_layer2.backward_layer.cell, ncps.keras.CfCCell)
