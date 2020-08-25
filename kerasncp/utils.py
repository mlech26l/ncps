# Copyright 2020 Mathias Lechner
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

from packaging.version import parse

try:
    import tensorflow as tf
except:
    raise ImportWarning(
        "It seems like the Tensorflow package is not installed\n"
        "Please run"
        "`$ pip install tensorflow`. \n",
    )


def check_tf_version():
    if parse(tf.__version__) < parse("2.0.0"):
        raise ImportError(
            "The Tensorflow package version needs to be at least 2.0.0 \n"
            "for keras-ncp to run. Currently, your TensorFlow version is \n"
            "{version}. Please upgrade with \n"
            "`$ pip install --upgrade tensorflow`. \n"
            "You can use `pip freeze` to check afterwards that everything is "
            "ok.".format(version=tf.__version__)
        )

