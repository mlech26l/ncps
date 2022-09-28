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
from kerasncp.torch.experimental import CfC


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
    assert output.size() == (3, 32)

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