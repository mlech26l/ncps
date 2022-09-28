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

import kerasncp
from kerasncp.tf.experimental import CfCCell, MixedMemoryRNN
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="kerasncp", name="CfC")
class CfC(tf.keras.layers.RNN):
    def __init__(
        self,
        wiring_or_units,
        mixed_memory=False,
        mode="default",
        activation="lecun_tanh",
        backbone_units=None,
        backbone_layers=None,
        backbone_dropout=None,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        time_major=False,
        **kwargs,
    ):
        if isinstance(wiring_or_units, kerasncp.wirings.Wiring):
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
                wiring_or_units,
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