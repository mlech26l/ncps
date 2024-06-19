# Copyright 2022 Mathias Lechner. All rights reserved
import numpy

from .cfc_cell import lecun_tanh, CfCCell

import keras
from ncps.wirings import wirings
import numpy as np


def split_tensor(input_tensor, num_or_size_splits, axis=0):
    """
    Splits the input tensor along the specified axis into multiple sub-tensors.

    Args:
        input_tensor (Tensor): The input tensor to be split.
        num_or_size_splits (int or list/tuple): If an integer, the number of equal splits along the axis.
                                                If a list/tuple, the sizes of each output tensor along the axis.
        axis (int): The axis along which to split the tensor. Default is 0.

    Returns:
        A list of tensors resulting from splitting the input tensor.
    """
    input_shape = keras.ops.shape(input_tensor)
    tensor_shape = input_shape[:axis] + (-1,) + input_shape[axis+1:]

    if isinstance(num_or_size_splits, int):
        split_sizes = [input_shape[axis] // num_or_size_splits] * num_or_size_splits
    else:
        split_sizes = num_or_size_splits

    split_tensors = []
    start = 0
    for size in split_sizes:
        end = start + size
        tensor = keras.layers.Lambda(lambda x: x[:, start:end], output_shape=tensor_shape)(input_tensor)
        split_tensors.append(tensor)
        start = end

    return split_tensors


@keras.utils.register_keras_serializable(package="ncps", name="WiredCfCCell")
class WiredCfCCell(keras.layers.Layer):
    def __init__(
        self,
        wiring,
        fully_recurrent=True,
        mode="default",
        activation="lecun_tanh",
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        return self._wiring.units
        # return [
        #     len(self._wiring.get_neurons_of_layer(i))
        #     for i in range(self._wiring.num_layers)
        # ]

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
            cell = CfCCell(
                len(layer_i_neurons),
                input_sparsity,
                recurrent_sparsity,
                mode=self.mode,
                activation=self._activation,
                backbone_units=0,
                backbone_layers=0,
                backbone_dropout=0,
            )

            cell_in_shape = (None, input_sparsity.shape[0])
            # cell.build(cell_in_shape)
            self._cfc_layers.append(cell)

        self._layer_sizes = [l.units for l in self._cfc_layers]
        self.built = True

    def call(self, inputs, states, **kwargs):
        if isinstance(inputs, (tuple, list)):
            # Irregularly sampled mode
            inputs, t = inputs
            t = keras.ops.reshape(t, [-1, 1])
        else:
            # Regularly sampled mode (elapsed time = 1 second)
            t = 1.0

        states = split_tensor(states[0], self._layer_sizes, axis=-1)
        assert len(states) == self._wiring.num_layers, \
            f'Incompatible num of states [{len(states)}] and wiring layers [{self._wiring.num_layers}]'
        new_hiddens = []
        for i, cfc_layer in enumerate(self._cfc_layers):
            if t == 1.0:
                output, new_hidden = cfc_layer(inputs, [states[i]], time=t)
            else:
                output, new_hidden = cfc_layer((inputs, t), [states[i]])
            cfc_layer._allow_non_tensor_positional_args = True
            new_hiddens.append(new_hidden[0])
            inputs = output

        assert len(new_hiddens) == self._wiring.num_layers, \
            f'Internal error new_hiddens [{new_hiddens}] != num_layers [{self._wiring.num_layers}]'
        if self._wiring.output_dim != output.shape[-1]:
            output = output[:, 0: self._wiring.output_dim]

        new_hiddens = keras.ops.concatenate(new_hiddens, axis=-1)
        return output, new_hiddens

    def get_config(self):
        seralized = self._wiring.get_config()
        seralized["mode"] = self.mode
        seralized["activation"] = self._activation
        seralized["backbone_units"] = None
        seralized["backbone_layers"] = None
        seralized["backbone_dropout"] = None
        return seralized

    @classmethod
    def from_config(cls, config):
        wiring = wirings.Wiring.from_config(config)
        return cls(wiring=wiring, **config)
