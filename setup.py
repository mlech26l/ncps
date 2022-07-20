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

from __future__ import absolute_import
from setuptools import setup, find_packages

setup(
    name="keras-ncp",
    version="2.0.11",
    packages=find_packages(),  # include/exclude arguments take * as wildcard, . for any sub-package names
    description="Neural Circuit Policies implementation for Keras (TensorFlow 2) and PyTorch",
    url="https://github.com/mlech26l/keras-ncp",
    author="Mathias Lechner",
    author_email="mlech26l@gmail.com",
    license="Apache License 2.0",
    # tensorflow and torch isn't a dependency because it would force the
    # download of the gpu version or the cpu version.
    # users should install it manually.
    install_requires=[
        "packaging",
        "future",
        "networkx",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
)
