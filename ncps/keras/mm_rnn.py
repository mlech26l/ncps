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


import keras


@keras.utils.register_keras_serializable(package="ncps", name="MixedMemoryRNN")
class MixedMemoryRNN(keras.layers.Layer):
    def __init__(self, rnn_cell, forget_gate_bias=1.0, **kwargs):
        super().__init__(**kwargs)

        self.rnn_cell = rnn_cell
        self.forget_gate_bias = forget_gate_bias

    @property
    def state_size(self):
        return [self.flat_size, self.rnn_cell.state_size]

    @property
    def flat_size(self):
        if isinstance(self.rnn_cell.state_size, int):
            return self.rnn_cell.state_size
        return sum(self.rnn_cell.state_size)

    def build(self, sequences_shape, initial_state_shape=None):
        input_dim = sequences_shape[-1]
        if isinstance(sequences_shape[0], tuple) or isinstance(sequences_shape[0], keras.KerasTensor):
            # Nested tuple
            input_dim = sequences_shape[0][-1]

        self.rnn_cell.build((None, self.flat_size))
        self.input_kernel = self.add_weight(
            shape=(input_dim, 4 * self.flat_size),
            initializer="glorot_uniform",
            name="input_kernel",
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.flat_size, 4 * self.flat_size),
            initializer="orthogonal",
            name="recurrent_kernel",
        )
        self.bias = self.add_weight(
            shape=(4 * self.flat_size,),
            initializer=keras.initializers.Zeros(),
            name="bias",
        )

        self.built = True

    def call(self, sequences, initial_state=None, mask=None, training=False, **kwargs):
        memory_state, ct_state = initial_state
        flat_ct_state = keras.ops.concatenate([ct_state], axis=-1)
        z = (
                keras.ops.matmul(sequences, self.input_kernel)
                + keras.ops.matmul(flat_ct_state, self.recurrent_kernel)
                + self.bias
        )

        i, ig, fg, og = keras.ops.split(z, 4, axis=-1)

        input_activation = keras.activations.tanh(i)
        input_gate = keras.activations.sigmoid(ig)
        forget_gate = keras.activations.sigmoid(fg + self.forget_gate_bias)
        output_gate = keras.activations.sigmoid(og)

        new_memory_state = memory_state * forget_gate + input_activation * input_gate
        ct_input = keras.activations.tanh(new_memory_state) * output_gate  # LSTM output = ODE input

        if (isinstance(sequences, tuple) or isinstance(sequences, list)) and len(sequences) > 1:
            # Input is a tuple -> Ct cell input should also be a tuple
            ct_input = (ct_input,) + sequences[1:]

        # Implementation choice on how to parametrize ODE component
        if (not isinstance(ct_state, tuple)) and (not isinstance(ct_state, list)):
            ct_state = [ct_state]

        ct_output, new_ct_state = self.rnn_cell(ct_input, ct_state, **kwargs)

        return ct_output, [new_memory_state, new_ct_state]

    def get_config(self):
        serialized = {
            "rnn_cell": self.rnn_cell.get_config(),
            "forget_gate_bias": self.forget_gate_bias,
        }
        return serialized

    @classmethod
    def from_config(cls, config, custom_objects=None):
        rnn_cell = keras.layers.deserialize(config["rnn_cell"])
        return cls(rnn_cell=rnn_cell, **config)
