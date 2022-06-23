# Copyright 2020-2021 Mathias Lechner
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
import kerasncp.wirings
from kerasncp import wirings
import numpy as np
from packaging.version import parse

try:
    import tensorflow as tf
except:
    raise ImportWarning(
        "It seems like the Tensorflow package is not installed\n"
        "Please run"
        "`$ pip install tensorflow`. \n",
    )

if parse(tf.__version__) < parse("2.0.0"):
    raise ImportError(
        "The Tensorflow package version needs to be at least 2.0.0 \n"
        "for keras-ncp to run. Currently, your TensorFlow version is \n"
        "{version}. Please upgrade with \n"
        "`$ pip install --upgrade tensorflow`. \n"
        "You can use `pip freeze` to check afterwards that everything is "
        "ok.".format(version=tf.__version__)
    )


# LeCun improved tanh activation
# http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf


def lecun_tanh(x):
    return 1.7159 * tf.nn.tanh(0.666 * x)


@tf.keras.utils.register_keras_serializable(package="Custom", name="CfcCell")
class CfcCell(tf.keras.layers.AbstractRNNCell):
    def __init__(
        self,
        units,
        input_sparsity=None,
        recurrent_sparsity=None,
        mode="default",
        activation="lecun_tanh",
        hidden_units=128,
        hidden_layers=1,
        hidden_dropout=0.1,
        **kwargs,
    ):
        super(CfcCell, self).__init__(**kwargs)
        if isinstance(units, kerasncp.wirings.Wiring):
            raise ValueError(
                "For defining sparse Cfc networks use WiredCfcCell instead of CfcCell"
            )
        self.units = units
        self.sparsity_mask = None
        if input_sparsity is not None or recurrent_sparsity is not None:
            # No backbone is allowed
            if hidden_units > 0:
                raise ValueError(
                    "If sparsity of a Cfc cell is set, then no backbone is allowed"
                )
            # Both need to be set
            if input_sparsity is None or recurrent_sparsity is None:
                raise ValueError(
                    "If sparsity of a Cfc cell is set, then both input and recurrent sparsity needs to be defined"
                )
            self.sparsity_mask = tf.constant(
                np.concatenate([input_sparsity, recurrent_sparsity], axis=0),
                dtype=tf.float32,
            )

        allowed_modes = ["default", "pure", "no_gate"]
        if mode not in allowed_modes:
            raise ValueError(
                "Unknown mode '{}', valid options are {}".format(
                    mode, str(allowed_modes)
                )
            )
        self.mode = mode
        self.backbone_fn = None
        if activation == "lecun_tanh":
            activation = lecun_tanh
        self._activation = activation
        self._hidden_units = hidden_units
        self._hidden_layers = hidden_layers
        self._hidden_dropout = hidden_dropout
        self._cfc_layers = []

    @property
    def state_size(self):
        return self.units

    def build(self, input_shape):
        if isinstance(input_shape[0], tuple) or isinstance(
            input_shape[0], tf.TensorShape
        ):
            # Nested tuple -> First item represent feature dimension
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]

        backbone_layers = []
        for i in range(self._hidden_layers):

            backbone_layers.append(
                tf.keras.layers.Dense(
                    self._hidden_units, self._activation, name=f"backbone{i}"
                )
            )
            backbone_layers.append(tf.keras.layers.Dropout(self._hidden_dropout))

        self.backbone_fn = tf.keras.models.Sequential(backbone_layers)
        cat_shape = int(
            self.state_size + input_dim
            if self._hidden_layers == 0
            else self._hidden_units
        )
        if self.mode == "pure":
            self.ff1_kernel = self.add_weight(
                shape=(cat_shape, self.state_size),
                initializer="glorot_uniform",
                name="ff1_weight",
            )
            self.ff1_bias = self.add_weight(
                shape=(self.state_size,),
                initializer="zeros",
                name="ff1_bias",
            )
            self.w_tau = self.add_weight(
                shape=(1, self.state_size),
                initializer=tf.keras.initializers.Zeros(),
                name="w_tau",
            )
            self.A = self.add_weight(
                shape=(1, self.state_size),
                initializer=tf.keras.initializers.Ones(),
                name="A",
            )
        else:
            self.ff1_kernel = self.add_weight(
                shape=(cat_shape, self.state_size),
                initializer="glorot_uniform",
                name="ff1_weight",
            )
            self.ff1_bias = self.add_weight(
                shape=(self.state_size,),
                initializer="zeros",
                name="ff1_bias",
            )
            self.ff2_kernel = self.add_weight(
                shape=(cat_shape, self.state_size),
                initializer="glorot_uniform",
                name="ff2_weight",
            )
            self.ff2_bias = self.add_weight(
                shape=(self.state_size,),
                initializer="zeros",
                name="ff2_bias",
            )

            #  = tf.keras.layers.Dense(
            #     , self._activation, name=f"{self.name}/ff1"
            # )
            # self.ff2 = tf.keras.layers.Dense(
            #     self.state_size, self._activation, name=f"{self.name}/ff2"
            # )
            # if self.sparsity_mask is not None:
            #     self.ff1.build((None,))
            #     self.ff2.build((None, self.sparsity_mask.shape[0]))
            self.time_a = tf.keras.layers.Dense(self.state_size, name="time_a")
            self.time_b = tf.keras.layers.Dense(self.state_size, name="time_b")
        self.built = True

    def call(self, inputs, states, **kwargs):
        if isinstance(inputs, (tuple, list)):
            # Irregularly sampled mode
            inputs, t = inputs
            t = tf.reshape(t, [-1, 1])
        else:
            # Regularly sampled mode (elapsed time = 1 second)
            t = 1.0

        x = tf.keras.layers.Concatenate()([inputs, states[0]])
        x = self.backbone_fn(x)
        if self.sparsity_mask is not None:
            ff1_kernel = self.ff1_kernel * self.sparsity_mask
            ff1 = tf.matmul(x, ff1_kernel) + self.ff1_bias
        else:
            ff1 = tf.matmul(x, self.ff1_kernel) + self.ff1_bias
        if self.mode == "pure":
            # Solution
            new_hidden = (
                -self.A
                * tf.math.exp(-t * (tf.math.abs(self.w_tau) + tf.math.abs(ff1)))
                * ff1
                + self.A
            )
        else:
            # Cfc
            if self.sparsity_mask is not None:
                ff2_kernel = self.ff2_kernel * self.sparsity_mask
                ff2 = tf.matmul(x, ff2_kernel) + self.ff2_bias
            else:
                ff2 = tf.matmul(x, self.ff2_kernel) + self.ff2_bias
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            t_interp = tf.nn.sigmoid(-t_a * t + t_b)
            if self.mode == "no_gate":
                new_hidden = ff1 + t_interp * ff2
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2

        return new_hidden, [new_hidden]


@tf.keras.utils.register_keras_serializable(package="Custom", name="WiredCfcCell")
class WiredCfcCell(tf.keras.layers.AbstractRNNCell):
    def __init__(
        self,
        wiring,
        fully_recurrent=True,
        mode="default",
        activation="lecun_tanh",
        **kwargs,
    ):
        super(WiredCfcCell, self).__init__(**kwargs)
        self._wiring = wiring
        allowed_modes = ["default", "pure", "no_gate"]
        if mode not in allowed_modes:
            raise ValueError(
                "Unknown mode '{}', valid options are {}".format(
                    mode, str(allowed_modes)
                )
            )
        self.mode = mode
        self.fully_recurrent = fully_recurrent
        if activation == "lecun_tanh":
            activation = lecun_tanh
        self._activation = activation
        self._cfc_layers = []

    @property
    def state_size(self):
        return [
            len(self._wiring.get_neurons_of_layer(i))
            for i in range(self._wiring.num_layers)
        ]

    @property
    def input_size(self):
        return self._wiring.input_dim

    def build(self, input_shape):
        if isinstance(input_shape[0], tuple):
            # Nested tuple -> First item represent feature dimension
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]

        self._wiring.build(input_dim)
        for i in range(self._wiring.num_layers):
            layer_i_neurons = self._wiring.get_neurons_of_layer(i)
            if i == 0:
                input_sparsity = self._wiring.sensory_adjacency_matrix[
                    :, layer_i_neurons
                ]
            else:
                prev_layer_neurons = self._wiring.get_neurons_of_layer(i - 1)
                input_sparsity = self._wiring.adjacency_matrix[:, layer_i_neurons]
                input_sparsity = input_sparsity[prev_layer_neurons, :]
            if self.fully_recurrent:
                recurrent_sparsity = np.ones(
                    (len(layer_i_neurons), len(layer_i_neurons)), dtype=np.int32
                )
            else:
                recurrent_sparsity = self._wiring.adjacency_matrix[
                    layer_i_neurons, layer_i_neurons
                ]
            cell = CfcCell(
                len(layer_i_neurons),
                input_sparsity,
                recurrent_sparsity,
                mode=self.mode,
                activation=self._activation,
                hidden_units=0,
                hidden_layers=0,
                hidden_dropout=0,
            )

            cell_in_shape = (None, input_sparsity.shape[0])
            # cell.build(cell_in_shape)
            self._cfc_layers.append(cell)

        self.built = True

    def call(self, inputs, states, **kwargs):
        if isinstance(inputs, (tuple, list)):
            # Irregularly sampled mode
            inputs, t = inputs
            t = tf.reshape(t, [-1, 1])
        else:
            # Regularly sampled mode (elapsed time = 1 second)
            t = 1.0

        assert len(states) == self._wiring.num_layers
        new_hiddens = []
        for i, layer in enumerate(self._cfc_layers):
            layer_input = (inputs, t)
            output, new_hidden = layer(layer_input, [states[i]])
            new_hiddens.append(new_hidden[0])
            inputs = output

        assert len(new_hiddens) == self._wiring.num_layers
        if self._wiring.output_dim != output.shape[-1]:
            output = output[:, 0 : self._wiring.output_dim]
        return output, new_hiddens

    def get_config(self):
        seralized = self._wiring.get_config()
        seralized["mode"] = self.mode
        seralized["activation"] = self._activation
        seralized["hidden_units"] = self.hidden_units
        seralized["hidden_layers"] = self.hidden_layers
        seralized["hidden_dropout"] = self.hidden_dropout
        return seralized

    @classmethod
    def from_config(cls, config):
        wiring = wirings.Wiring.from_config(config)
        return cls(wiring=wiring, **config)
