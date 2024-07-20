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

import ncps
import keras
from . import LTCCell, MixedMemoryRNN


@keras.utils.register_keras_serializable(package="ncps", name="LTC")
class LTC(keras.layers.RNN):
    name = "LTC"

    def __init__(
            self,
            units,
            mixed_memory: bool = False,
            input_mapping="affine",
            output_mapping="affine",
            ode_unfolds=6,
            epsilon=1e-8,
            initialization_ranges=None,
            return_sequences: bool = False,
            return_state: bool = False,
            go_backwards: bool = False,
            stateful: bool = False,
            unroll: bool = False,
            zero_output_for_mask: bool = False,
            **kwargs,
    ):
        """Applies a `Liquid time-constant (LTC) <https://ojs.aaai.org/index.php/AAAI/article/view/16936>`_ RNN to an input sequence.

        Examples::

            >>> from ncps.keras import LTC
            >>>
            >>> rnn = LTC(50)
            >>> x = keras.random.uniform((2, 10, 20))  # (B,L,C)
            >>> y = rnn(x)

        .. Note::
            For creating a wired `Neural circuit policy (NCP) <https://publik.tuwien.ac.at/files/publik_292280.pdf>`_ you can pass a `ncps.wirings.NCP` object instead of the number of units

        Examples::

            >>> from ncps.keras import LTC
            >>> from ncps.wirings import NCP
            >>>
            >>> wiring = NCP(10, 10, 8, 6, 6, 4, 4)
            >>> rnn = LTC(wiring)
            >>> x = keras.random.uniform((2, 10, 20))  # (B,L,C)
            >>> y = rnn(x)

        :param units: Wiring (ncps.wirings.Wiring instance) or integer representing the number of (fully-connected) hidden units
        :param mixed_memory: Whether to augment the RNN by a `memory-cell <https://arxiv.org/abs/2006.04418>`_ to help learn long-term dependencies in the data
        :param input_mapping: Mapping applied to the sensory neurons. Possible values None, "linear", "affine" (default "affine")
        :param output_mapping: Mapping applied to the motor neurons. Possible values None, "linear", "affine" (default "affine")
        :param ode_unfolds: Number of ODE-solver steps per time-step (default 6)
        :param epsilon: Auxillary value to avoid dividing by 0 (default 1e-8)
        :param initialization_ranges: A dictionary for overwriting the range of the uniform weight initialization (default None)
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
            self.wiring = units
        else:
            self.wiring = ncps.wirings.FullyConnected(units)

        cell = LTCCell(
            wiring=self.wiring,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            ode_unfolds=ode_unfolds,
            epsilon=epsilon,
            initialization_ranges=initialization_ranges,
            **kwargs,
        )
        if mixed_memory:
            cell = MixedMemoryRNN(cell)
        super(LTC, self).__init__(
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
        cell: LTCCell = self.cell.rnn_cell if is_mixed_memory else self.cell
        cell_config = cell.get_config()
        config = super(LTC, self).get_config()
        config["units"] = self.wiring
        config["mixed_memory"] = is_mixed_memory
        return {**cell_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # The following parameters are recreated by the LTC constructor
        del config["cell"]
        del config["wiring"]
        units = ncps.wirings.Wiring.from_config(config["units"]["config"])
        del config["units"]
        return cls(units, **config)
