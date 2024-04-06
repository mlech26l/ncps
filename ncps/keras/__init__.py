# Copyright 2020-2021 Mathias Lechner
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


from __future__ import absolute_import

from .ltc_cell import LTCCell
from .mm_rnn import MixedMemoryRNN
from .cfc_cell import CfCCell
from .wired_cfc_cell import WiredCfCCell
from .cfc import CfC
from .ltc import LTC
from packaging.version import parse

try:
    import keras
except:
    raise ImportWarning(
        "It seems like the Keras package is not installed\n"
        "Please run"
        "`$ pip install keras`. \n",
    )

if parse(keras.__version__) < parse("3.0.0"):
    raise ImportError(
        "The Keras package version needs to be at least 3.0.0 \n"
        "for ncps-keras to run. Currently, your Keras version is \n"
        "{version}. Please upgrade with \n"
        "`$ pip install --upgrade keras`. \n"
        "You can use `pip freeze` to check afterwards that everything is "
        "ok.".format(version=keras.__version__)
    )
__all__ = ["CfC", "CfCCell", "LTC", "LTCCell", "MixedMemoryRNN", "WiredCfCCell"]
