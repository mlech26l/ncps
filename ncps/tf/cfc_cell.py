# Copyright 2022 Mathias Lechner and Ramin Hasani
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


import numpy as np
import tensorflow as tf
from typing import Optional, Union


# LeCun improved tanh activation
# http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
def lecun_tanh(x):
    return 1.7159 * tf.nn.tanh(0.666 * x)


@tf.keras.utils.register_keras_serializable(package="ncps", name="CfCCell")
class CfCCell(tf.keras.layers.AbstractRNNCell):
    def __init__(
        self,
        units,
        input_sparsity=None,
        recurrent_sparsity=None,
        mode="default",
        activation="lecun_tanh",
        backbone_units=128,
        backbone_layers=1,
        backbone_dropout=0.1,
        **kwargs,
    ):
        """A `Closed-form Continuous-time <https://arxiv.org/abs/2106.13898>`_ cell.

        .. Note::
            This is an RNNCell that process single time-steps.
            To get a full RNN that can process sequences,
            see `ncps.tf.CfC` or wrap the cell with a `tf.keras.layers.RNN <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN>`_.


        :param units: Number of hidden units
        :param input_sparsity:
        :param recurrent_sparsity:
        :param mode: Either "default", "pure" (direct solution approximation), or "no_gate" (without second gate).
        :param activation: Activation function used in the backbone layers
        :param backbone_units: Number of hidden units in the backbone layer (default 128)
        :param backbone_layers: Number of backbone layers (default 1)
        :param backbone_dropout: Dropout rate in the backbone layers (default 0)
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.units = units
        self.sparsity_mask = None
        if input_sparsity is not None or recurrent_sparsity is not None:
            # No backbone is allowed
            if backbone_units > 0:
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
        self._backbone_units = backbone_units
        self._backbone_layers = backbone_layers
        self._backbone_dropout = backbone_dropout
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
        for i in range(self._backbone_layers):
            backbone_layers.append(
                tf.keras.layers.Dense(
                    self._backbone_units, self._activation, name=f"backbone{i}"
                )
            )
            backbone_layers.append(tf.keras.layers.Dropout(self._backbone_dropout))

        self.backbone_fn = tf.keras.models.Sequential(backbone_layers)
        cat_shape = int(
            self.state_size + input_dim
            if self._backbone_layers == 0
            else self._backbone_units
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
