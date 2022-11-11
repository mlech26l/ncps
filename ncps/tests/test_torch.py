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

import numpy as np
import sys
import time
import pytest
import torch
from ncps.torch import CfC, LTCCell, LTC
import ncps


def test_ncps():
    input_size = 8

    wiring = ncps.wirings.FullyConnected(8, 4)  # 16 units, 8 motor neurons
    ltc_cell = LTCCell(wiring, input_size)
    input = torch.randn(3, input_size)
    hx = torch.zeros(3, wiring.units)
    output, hx = ltc_cell(input, hx)
    assert output.size() == (3, 4)
    assert hx.size() == (3, wiring.units)


def test_default():
    input_size = 8
    hidden_size = 32
    rnn = CfC(input_size, hidden_size)
    input = torch.randn(5, 3, input_size)
    output, hx = rnn(input)
    assert output.size() == (5, 3, 32)


def test_batch_first():
    input_size = 8
    hidden_size = 32
    rnn = CfC(input_size, hidden_size, batch_first=False)
    input = torch.randn(5, 3, input_size)
    output, hx = rnn(input)
    assert output.size() == (5, 3, 32)


def test_proj():
    input_size = 8
    hidden_size = 32
    rnn = CfC(input_size, hidden_size, proj_size=10)
    input = torch.randn(5, 3, input_size)
    output, hx = rnn(input)
    assert output.size() == (5, 3, 10)


def test_unbatched_1():
    input_size = 8
    hidden_size = 32
    rnn = CfC(input_size, hidden_size, batch_first=True)
    input = torch.randn(3, input_size)
    output, hx = rnn(input)
    assert output.size() == (3, 32)


def test_unbatched_2():
    input_size = 8
    hidden_size = 32
    rnn = CfC(input_size, hidden_size, batch_first=False)
    input = torch.randn(3, input_size)
    output, hx = rnn(input)
    assert output.size() == (3, hidden_size)


def test_ltc_1():
    input_size = 8
    hidden_size = 32
    rnn = LTC(input_size, hidden_size, batch_first=True)
    input = torch.randn(5, 3, input_size)
    output, hx = rnn(input)
    assert output.size() == (5, 3, hidden_size)


def test_ltc_batch_first():
    input_size = 8
    hidden_size = 32
    rnn = LTC(input_size, hidden_size, batch_first=False)
    input = torch.randn(5, 3, input_size)
    output, hx = rnn(input)
    assert output.size() == (5, 3, 32)
    assert hx.size() == (3, 32)


def test_ncp_1():
    input_size = 8
    wiring = ncps.wirings.NCP(10, 10, 8, 6, 6, 4, 6)
    rnn = LTC(input_size, wiring, batch_first=True)
    assert wiring.synapse_count > 0
    assert wiring.sensory_synapse_count > 0
    input = torch.randn(5, 3, input_size)
    output, hx = rnn(input)
    assert output.size() == (5, 3, 8)
    assert hx.size() == (5, 10 + 10 + 8)


def test_auto_ncp_1():
    input_size = 8
    wiring = ncps.wirings.AutoNCP(16, 4)
    rnn = LTC(input_size, wiring, batch_first=True)
    input = torch.randn(5, 3, input_size)
    output, hx = rnn(input)
    assert output.size() == (5, 3, 4)
    assert hx.size() == (5, 16)


def test_ncp_2():
    input_size = 8
    wiring = ncps.wirings.NCP(10, 10, 8, 6, 6, 4, 6)
    rnn = LTC(input_size, wiring, batch_first=False)
    input = torch.randn(5, 3, input_size)
    output, hx = rnn(input)
    assert output.size() == (5, 3, 8)
    assert hx.size() == (3, 10 + 10 + 8)


def test_ncp_cfc_1():
    input_size = 8
    wiring = ncps.wirings.NCP(10, 10, 8, 6, 6, 4, 6)
    rnn = CfC(input_size, wiring, batch_first=True)
    input = torch.randn(5, 3, input_size)
    output, hx = rnn(input)
    assert output.size() == (5, 3, 8)
    assert hx.size() == (5, 10 + 10 + 8)


def test_auto_ncp_cfc_1():
    input_size = 8
    wiring = ncps.wirings.AutoNCP(28, 10)
    rnn = CfC(input_size, wiring, batch_first=True)
    input = torch.randn(5, 3, input_size)
    output, hx = rnn(input)
    assert output.size() == (5, 3, 10)
    assert hx.size() == (5, 28)


def test_ncp_cfc_2():
    input_size = 8
    wiring = ncps.wirings.NCP(10, 10, 8, 6, 6, 4, 6)
    rnn = CfC(input_size, wiring, batch_first=False)
    input = torch.randn(5, 3, input_size)
    output, hx = rnn(input)
    assert output.size() == (5, 3, 8)
    assert hx.size() == (3, 10 + 10 + 8)

    # def __init__(
    #         self,
    #         input_size,
    #         hidden_size,
    #         proj_size=None,
    #         return_sequences=True,
    #         batch_first=True,
    #         use_mm_rnn=False,
    #         mode="default",
    #         backbone_activation="lecun",
    #         backbone_units=128,
    #         backbone_layers=1,
    #         backbone_dr=0.0,
    # ):


if __name__ == "__main__":
    import traceback
    import warnings
    import sys

    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
        log = file if hasattr(file, "write") else sys.stderr
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = warn_with_traceback
    test_ncp_cfc_2()