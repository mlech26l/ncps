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

import os

import numpy as np
import hashlib
import urllib.request


def _augment_data(data):
    new_data = []
    for x, y in data:
        new_data.append((x, y))
        x2 = x[:, ::-1]  # Mirror lidar signal (reverse on axis 1)
        y2 = -y  # Invert steering command
        new_data.append((x2, y2))
    return new_data


def _unpack(path):
    all_files = []
    with np.load(path) as f:
        for i in range(29):  # there should be exactly 29 files in the npz
            all_files.append((f["x_{}".format(i)], f["y_{}".format(i)]))

    return all_files


def _train_test_split(all_files):
    np.random.default_rng(20200822).shuffle(all_files)
    # Use 7 files for test (~ 25%)
    test_files = all_files[:7]
    train_files = all_files[7:]

    return train_files, test_files


def _align_in_sequences(data, seq_len):
    aligned_x, aligned_y = [], []
    for x, y in data:
        # Iterate over file with 75% overlap
        for i in range(0, x.shape[0] - seq_len, seq_len // 2):
            # Cut out sequence of exactly ''seq_len'' length
            aligned_x.append(x[i : i + seq_len])
            aligned_y.append(y[i : i + seq_len])

    aligned_x = np.stack(aligned_x, axis=0)
    aligned_y = np.stack(aligned_y, axis=0)

    # Let's add a extra axis at the end to make life simpler for working with a
    # 1D ConvNet
    aligned_x = np.expand_dims(aligned_x, axis=-1)
    aligned_y = np.expand_dims(aligned_y, axis=-1)

    return (aligned_x, aligned_y)


def load_data(local_path=None, seq_len=32):
    url = "https://github.com/mlech26l/icra_lds/raw/master/icra2020_imitation_data_packed.npz"
    if local_path is None:
        os.makedirs("datasets", exist_ok=True)
        local_path = os.path.join("datasets", "icra2020_imitation_data_packed.npz")
    download_file = True
    if os.path.isfile(local_path):
        md5 = hashlib.md5(open(local_path, "rb").read()).hexdigest()
        if md5 == "15ab035e0866fc065acfc0ad781d75c5":
            # No need to download file
            download_file = False
    if download_file:
        print("Downloading file '{}'".format(url))
        with urllib.request.urlopen(url) as response, open(
            local_path, "wb"
        ) as out_file:
            data = response.read()
            out_file.write(data)

    data = _unpack(local_path)
    train_data, test_data = _train_test_split(data)

    train_data = _augment_data(train_data)
    test_data = _augment_data(test_data)

    train_data = _align_in_sequences(train_data, seq_len)
    test_data = _align_in_sequences(test_data, seq_len)

    return (train_data, test_data)
