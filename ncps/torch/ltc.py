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
import torch
from torch import nn
from typing import Optional, Union
import ncps
from . import CfCCell, LTCCell
from .lstm import LSTMCell


class LTC(nn.Module):
    def __init__(
        self,
        input_size: int,
        units,
        return_sequences: bool = True,
        batch_first: bool = True,
        mixed_memory: bool = False,
        input_mapping="affine",
        output_mapping="affine",
        ode_unfolds=6,
        epsilon=1e-8,
        implicit_param_constraints=True,
    ):
        """Applies a `Liquid time-constant (LTC) <https://ojs.aaai.org/index.php/AAAI/article/view/16936>`_ RNN to an input sequence.

        Examples::

             >>> from ncps.torch import LTC
             >>>
             >>> rnn = LTC(20,50)
             >>> x = torch.randn(2, 3, 20) # (batch, time, features)
             >>> h0 = torch.zeros(2,50) # (batch, units)
             >>> output, hn = rnn(x,h0)

        .. Note::
            For creating a wired `Neural circuit policy (NCP) <https://publik.tuwien.ac.at/files/publik_292280.pdf>`_ you can pass a `ncps.wirings.NCP` object instead of the number of units

        Examples::

             >>> from ncps.torch import LTC
             >>> from ncps.wirings import NCP
             >>>
             >>> wiring = NCP(10, 10, 8, 6, 6, 4, 6)
             >>> rnn = LTC(20, wiring)

             >>> x = torch.randn(2, 3, 20) # (batch, time, features)
             >>> h0 = torch.zeros(2, 28) # (batch, units)
             >>> output, hn = rnn(x,h0)


        :param input_size: Number of input features
        :param units: Wiring (ncps.wirings.Wiring instance) or integer representing the number of (fully-connected) hidden units
        :param return_sequences: Whether to return the full sequence or just the last output
        :param batch_first: Whether the batch or time dimension is the first (0-th) dimension
        :param mixed_memory: Whether to augment the RNN by a `memory-cell <https://arxiv.org/abs/2006.04418>`_ to help learn long-term dependencies in the data
        :param input_mapping:
        :param output_mapping:
        :param ode_unfolds:
        :param epsilon:
        :param implicit_param_constraints:
        """

        super(LTC, self).__init__()
        self.input_size = input_size
        self.wiring_or_units = units
        self.batch_first = batch_first
        self.return_sequences = return_sequences

        if isinstance(units, ncps.wirings.Wiring):
            wiring = units
        else:
            wiring = ncps.wirings.FullyConnected(units)
        self.rnn_cell = LTCCell(
            wiring=wiring,
            in_features=input_size,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            ode_unfolds=ode_unfolds,
            epsilon=epsilon,
            implicit_param_constraints=implicit_param_constraints,
        )
        self._wiring = wiring
        self.use_mixed = mixed_memory
        if self.use_mixed:
            self.lstm = LSTMCell(input_size, self.state_size)

    @property
    def state_size(self):
        return self._wiring.units

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

    def forward(self, input, hx=None, timespans=None):
        """

        :param input: Input tensor of shape (L,C) in batchless mode, or (B,L,C) if batch_first was set to True and (L,B,C) if batch_first is False
        :param hx: Initial hidden state of the RNN of shape (B,H) if mixed_memory is False and a tuple ((B,H),(B,H)) if mixed_memory is True. If None, the hidden states are initialized with all zeros.
        :param timespans:
        :return: A pair (output, hx), where output and hx the final hidden state of the RNN
        """
        device = input.device
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0
        if not is_batched:
            input = input.unsqueeze(batch_dim)
            if timespans is not None:
                timespans = timespans.unsqueeze(batch_dim)

        batch_size, seq_len = input.size(batch_dim), input.size(seq_dim)

        if hx is None:
            h_state = torch.zeros((batch_size, self.state_size), device=device)
            c_state = (
                torch.zeros((batch_size, self.state_size), device=device)
                if self.use_mixed
                else None
            )
        else:
            if self.use_mixed and isinstance(hx, torch.Tensor):
                raise RuntimeError(
                    "Running a CfC with mixed_memory=True, requires a tuple (h0,c0) to be passed as state (got torch.Tensor instead)"
                )
            h_state, c_state = hx if self.use_mixed else (hx, None)
            if is_batched:
                if h_state.dim() != 2:
                    msg = (
                        "For batched 2-D input, hx and cx should "
                        f"also be 2-D but got ({h_state.dim()}-D) tensor"
                    )
                    raise RuntimeError(msg)
            else:
                # batchless  mode
                if h_state.dim() != 1:
                    msg = (
                        "For unbatched 1-D input, hx and cx should "
                        f"also be 1-D but got ({h_state.dim()}-D) tensor"
                    )
                    raise RuntimeError(msg)
                h_state = h_state.unsqueeze(0)
                c_state = c_state.unsqueeze(0) if c_state is not None else None

        output_sequence = []
        for t in range(seq_len):
            if self.batch_first:
                inputs = input[:, t]
                ts = 1.0 if timespans is None else timespans[:, t].squeeze()
            else:
                inputs = input[t]
                ts = 1.0 if timespans is None else timespans[t].squeeze()

            if self.use_mixed:
                h_state, c_state = self.lstm(inputs, (h_state, c_state))
            h_out, h_state = self.rnn_cell.forward(inputs, h_state, ts)
            if self.return_sequences:
                output_sequence.append(h_out)

        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            readout = torch.stack(output_sequence, dim=stack_dim)
        else:
            readout = h_out
        hx = (h_state, c_state) if self.use_mixed else h_state

        if not is_batched:
            # batchless  mode
            readout = readout.squeeze(batch_dim)
            hx = (h_state[0], c_state[0]) if self.use_mixed else h_state[0]

        return readout, hx