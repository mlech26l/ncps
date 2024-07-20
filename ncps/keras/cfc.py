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

from typing import Union

import keras

import ncps
from . import CfCCell, MixedMemoryRNN, WiredCfCCell


@keras.utils.register_keras_serializable(package="ncps", name="CfC")
class CfC(keras.layers.RNN):
    def __init__(
            self,
            units: Union[int, ncps.wirings.Wiring],
            mixed_memory: bool = False,
            mode: str = "default",
            activation: str = "lecun_tanh",
            backbone_units: int = None,
            backbone_layers: int = None,
            backbone_dropout: float = None,
            sparsity_mask: keras.layers.Layer = None,
            fully_recurrent: bool = True,
            return_sequences: bool = False,
            return_state: bool = False,
            go_backwards: bool = False,
            stateful: bool = False,
            unroll: bool = False,
            zero_output_for_mask: bool = False,
            **kwargs,
    ):
        """Applies a `Closed-form Continuous-time <https://arxiv.org/abs/2106.13898>`_ RNN to an input sequence.

        Examples::

            >>> from ncps.keras import CfC
            >>>
            >>> rnn = CfC(50)
            >>> x = keras.random.uniform((2, 10, 20))  # (B,L,C)
            >>> y = rnn(x)

        :param units: Number of hidden units
        :param mixed_memory: Whether to augment the RNN by a `memory-cell <https://arxiv.org/abs/2006.04418>`_ to help learn long-term dependencies in the data (default False)
        :param mode: Either "default", "pure" (direct solution approximation), or "no_gate" (without second gate). (default "default)
        :param activation: Activation function used in the backbone layers (default "lecun_tanh")
        :param backbone_units: Number of hidden units in the backbone layer (default 128)
        :param backbone_layers: Number of backbone layers (default 1)
        :param backbone_dropout: Dropout rate in the backbone layers (default 0)
        :param sparsity_mask:
        :param fully_recurrent: Whether to apply a fully-connected sparsity_mask or use the adjacency_matrix. Evaluated only for WiredCfCCell. (default True)
        :param return_sequences: Whether to return the full sequence or just the last output (default False)
        :param return_state: Whether to return just the output of the RNN or a tuple (output, last_hidden_state) (default False)
        :param go_backwards: If True, the input sequence will be process from back to the front (default False)
        :param stateful: Whether to remember the last hidden state of the previous inference/training batch and use it as initial state for the next inference/training batch (default False)
        :param unroll: Whether to unroll the graph, i.e., may increase speed at the cost of more memory (default False)
        :param zero_output_for_mask: Whether the output should use zeros for the masked timesteps. (default False)
                                     Note that this field is only used when `return_sequences` is `True` and `mask` is provided.
                                     It can be useful if you want to reuse the raw output sequence of
                                     the RNN without interference from the masked timesteps, e.g.,
                                     merging bidirectional RNNs.
        :param kwargs:
        """

        if isinstance(units, ncps.wirings.Wiring):
            if backbone_units is not None:
                raise ValueError(f"Cannot use backbone_units in wired mode")
            if backbone_layers is not None:
                raise ValueError(f"Cannot use backbone_layers in wired mode")
            if backbone_dropout is not None:
                raise ValueError(f"Cannot use backbone_dropout in wired mode")
            cell = WiredCfCCell(units, mode=mode, activation=activation, fully_recurrent=fully_recurrent)
        else:
            backbone_units = 128 if backbone_units is None else backbone_units
            backbone_layers = 1 if backbone_layers is None else backbone_layers
            backbone_dropout = 0.0 if backbone_dropout is None else backbone_dropout
            cell = CfCCell(
                units,
                mode=mode,
                activation=activation,
                backbone_units=backbone_units,
                backbone_layers=backbone_layers,
                backbone_dropout=backbone_dropout,
                sparsity_mask=sparsity_mask,
            )
        if mixed_memory:
            cell = MixedMemoryRNN(cell)
        super(CfC, self).__init__(
            cell,
            return_sequences,
            return_state,
            go_backwards,
            stateful,
            unroll,
            zero_output_for_mask,
            **kwargs,
        )

    def get_config(self):
        is_mixed_memory = isinstance(self.cell, MixedMemoryRNN)
        cell: CfCCell | WiredCfCCell = self.cell.rnn_cell if is_mixed_memory else self.cell
        cell_config = cell.get_config()
        config = super(CfC, self).get_config()
        config["units"] = cell.wiring if isinstance(cell, WiredCfCCell) else cell.units
        config["mixed_memory"] = is_mixed_memory
        config["fully_recurrent"] = cell.fully_recurrent if isinstance(cell, WiredCfCCell) else True # If not WiredCfc it's ignored
        return {**cell_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # The following parameters are recreated by the LTC constructor
        del config["cell"]
        if "wiring" in config:
            del config["wiring"]
            units = ncps.wirings.Wiring.from_config(config["units"]["config"])
        else:
            units = config["units"]
        del config["units"]
        return cls(units, **config)
