# Copyright 2021 Mathias Lechner
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
import tensorflow as tf
import kerasncp as kncp
from kerasncp.tf import LTCCell

data_x = np.random.default_rng().normal(size=(100, 16, 10))
data_y = np.random.default_rng().normal(size=(100, 16, 1))
print("data_y.shape: ", str(data_y.shape))

arch = kncp.wirings.Random(32, 1, sparsity_level=0.5)  # 32 units, 1 motor neuron
rnn_cell = LTCCell(arch)


model = tf.keras.models.Sequential(
    [
        tf.keras.Input((None, 10)),
        tf.keras.layers.RNN(rnn_cell, return_sequences=True),
    ]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.MeanSquaredError()
)

model.fit(x=data_x, y=data_y, batch_size=25, epochs=20)
model.evaluate(x=data_x, y=data_y)

model.save("test.h5")

restored_model = tf.keras.models.load_model("test.h5")

restored_model.evaluate(x=data_x, y=data_y)
