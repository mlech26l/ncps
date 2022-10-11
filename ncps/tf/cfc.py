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

import ncps
from . import CfCCell, MixedMemoryRNN
import tensorflow as tf
from typing import Optional, Union


@tf.keras.utils.register_keras_serializable(package="ncps", name="CfC")
class CfC(tf.keras.layers.RNN):
    def __init__(
        self,
        units: int,
        mixed_memory: bool = False,
        mode: str = "default",
        activation: str = "lecun_tanh",
        backbone_units: int = None,
        backbone_layers: int = None,
        backbone_dropout: float = None,
        return_sequences: bool = False,
        return_state: bool = False,
        go_backwards: bool = False,
        stateful: bool = False,
        unroll: bool = False,
        time_major: bool = False,
        **kwargs,
    ):
        """Applies a `Closed-form Continuous-time <https://arxiv.org/abs/2106.13898>`_ RNN to an input sequence.

        Args:
            units: Number of hidden units
            mixed_memory: Whether to augment the RNN by a `memory-cell <https://arxiv.org/abs/2006.04418>`_ to help learn long-term dependencies in the data
            mode: Either "default", "pure" (direct solution approximation), or "no_gate" (without second gate).
            activation: Activation function used in the backbone layers
            backbone_units: Number of hidden units in the backbone layer (default 128)
            backbone_layers: Number of backbone layers (default 1)
            backbone_dropout: Dropout rate in the backbone layers (default 0)
            return_sequences: Whether to return the full sequence or just the last output
            return_state: Whether to return just the output of the RNN or a tuple (output, last_hidden_state)
            go_backwards:
            stateful: Whether to remember the last hidden state of the previous inference/training batch and use it as initial state for the next inference/training batch
            unroll:
            time_major: Whether the time or batch dimension is the first (0-th) dimension
            **kwargs:
        """
        if isinstance(units, ncps.wirings.Wiring):
            if backbone_units is not None:
                raise ValueError(f"Cannot use backbone_units in wired mode")
            if backbone_layers is not None:
                raise ValueError(f"Cannot use backbone_layers in wired mode")
            if backbone_dropout is not None:
                raise ValueError(f"Cannot use backbone_dropout in wired mode")
            raise NotImplementedError()
            # cell = WiredCfCCell(wiring_or_units, mode=mode, activation=activation)
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
            time_major,
        )