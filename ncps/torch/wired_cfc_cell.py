# Copyright 2022 Mathias Lechner. All rights reserved

import numpy as np
import torch
from torch import nn

from . import CfCCell
from typing import Optional, Union


class WiredCfCCell(nn.Module):
    def __init__(
        self,
        input_size,
        wiring,
        mode="default",
    ):
        super(WiredCfCCell, self).__init__()

        if input_size is not None:
            wiring.build(input_size)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown number of input features. Please pass the parameter 'input_size' or call the 'wiring.build()'."
            )
        self._wiring = wiring

        self._layers = []
        in_features = wiring.input_dim
        for l in range(wiring.num_layers):
            hidden_units = self._wiring.get_neurons_of_layer(l)
            if l == 0:
                input_sparsity = self._wiring.sensory_adjacency_matrix[:, hidden_units]
            else:
                prev_layer_neurons = self._wiring.get_neurons_of_layer(l - 1)
                input_sparsity = self._wiring.adjacency_matrix[:, hidden_units]
                input_sparsity = input_sparsity[prev_layer_neurons, :]
            input_sparsity = np.concatenate(
                [
                    input_sparsity,
                    np.ones((len(hidden_units), len(hidden_units))),
                ],
                axis=0,
            )

            # Hack: nn.Module registers child params in set_attribute
            rnn_cell = CfCCell(
                in_features,
                len(hidden_units),
                mode,
                backbone_activation="lecun_tanh",
                backbone_units=0,
                backbone_layers=0,
                backbone_dropout=0.0,
                sparsity_mask=input_sparsity,
            )
            self.register_module(f"layer_{l}", rnn_cell)
            self._layers.append(rnn_cell)
            in_features = len(hidden_units)

    @property
    def state_size(self):
        return self._wiring.units

    @property
    def layer_sizes(self):
        return [
            len(self._wiring.get_neurons_of_layer(i))
            for i in range(self._wiring.num_layers)
        ]

    @property
    def num_layers(self):
        return self._wiring.num_layers

    @property
    def sensory_size(self):
        return self._wiring.input_dim

    @property
    def motor_size(self):
        return self._wiring.output_dim

    @property
    def output_size(self):
        return self.motor_size

    @property
    def synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    @property
    def sensory_synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    def forward(self, input, hx, timespans):
        h_state = torch.split(hx, self.layer_sizes, dim=1)

        new_h_state = []
        inputs = input
        for i in range(self.num_layers):
            h, _ = self._layers[i].forward(inputs, h_state[i], timespans)
            inputs = h
            new_h_state.append(h)

        new_h_state = torch.cat(new_h_state, dim=1)
        return h, new_h_state