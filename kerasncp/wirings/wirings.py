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

import numpy as np
import tensorflow as tf


class Wiring:
    def __init__(self, units):
        self.units = units
        self.adjacency_matrix = np.zeros([units, units], dtype=np.int32)
        self.input_dim = None
        self.output_dim = None

    def build(self, input_shape):
        input_dim = int(input_shape[1])
        if not self.input_dim is None and self.input_dim != input_dim:
            raise ValueError(
                "Conflicting input dimensions provided. set_input_dim() was called with {} but actual input has dimension {}".format(
                    self.input_dim, input_dim
                )
            )
        if self.input_dim is None:
            self.set_input_dim(input_dim)

    def erev_initializer(self, shape, dtype=None):
        return tf.convert_to_tensor(self.adjacency_matrix, dtype=tf.float32)

    def sensory_erev_initializer(self, shape, dtype=None):
        return tf.convert_to_tensor(self.sensory_adjacency_matrix, dtype=tf.float32)

    def set_input_dim(self, input_dim):
        self.input_dim = input_dim
        self.sensory_adjacency_matrix = np.zeros(
            [input_dim, self.units], dtype=np.int32
        )

    def set_output_dim(self, output_dim):
        self.output_dim = output_dim

    # May be overwritten by child class
    def get_type_of_neuron(self, neuron_id):
        return "motor" if neuron_id < self.output_dim else "inter"

    def add_synapse(self, src, dest, polarity):
        if src < 0 or src >= self.units:
            raise ValueError(
                "Cannot add synapse originating in {} if cell has only {} units".format(
                    src, self.units
                )
            )
        if dest < 0 or dest >= self.units:
            raise ValueError(
                "Cannot add synapse feeding into {} if cell has only {} units".format(
                    dest, self.units
                )
            )
        if not polarity in [-1, 1]:
            raise ValueError(
                "Cannot add synapse with polarity {} (expected -1 or +1)".format(
                    polarity
                )
            )
        self.adjacency_matrix[src, dest] = polarity

    def add_sensory_synapse(self, src, dest, polarity):
        if self.input_dim is None:
            raise ValueError(
                "Cannot add sensory synapses before build() has been called!"
            )
        if src < 0 or src >= self.input_dim:
            raise ValueError(
                "Cannot add sensory synapse originating in {} if input has only {} features".format(
                    src, self.input_dim
                )
            )
        if dest < 0 or dest >= self.units:
            raise ValueError(
                "Cannot add synapse feeding into {} if cell has only {} units".format(
                    dest, self.units
                )
            )
        if not polarity in [-1, 1]:
            raise ValueError(
                "Cannot add synapse with polarity {} (expected -1 or +1)".format(
                    polarity
                )
            )
        self.sensory_adjacency_matrix[src, dest] = polarity


class FullyConnected(Wiring):
    def __init__(self, units, output_dim=None, erev_init_seed=1111):
        super(FullyConnected, self).__init__(units)
        if output_dim is None:
            output_dim = units
        self.set_output_dim(output_dim)
        self._rng = np.random.default_rng(erev_init_seed)
        for src in range(self.units):
            for dest in range(self.units):
                polarity = self._rng.choice([-1, 1, 1])
                self.add_synapse(src, dest, polarity)

    def build(self, input_shape):
        super().build(input_shape)
        for src in range(self.input_dim):
            for dest in range(self.units):
                polarity = self._rng.choice([-1, 1, 1])
                self.add_sensory_synapse(src, dest, polarity)


class Random(Wiring):
    def __init__(self, units, output_dim=None, sparsity_level=0.0, random_seed=1111):
        super(Random, self).__init__(units)
        if output_dim is None:
            output_dim = units
        self.set_output_dim(output_dim)
        self.sparsity_level = sparsity_level

        if sparsity_level < 0.0 or sparsity_level >= 1.0:
            raise ValueError(
                "Invalid sparsity level '{}', expected value in range [0,1)".format(
                    sparsity_level
                )
            )
        self._rng = np.random.default_rng(random_seed)

        number_of_synapses = int(np.round(units * units * (1 - sparsity_level)))
        all_synapses = []
        for src in range(self.units):
            for dest in range(self.units):
                all_synapses.append((src, dest))

        used_syanpses = self._rng.choice(
            all_synapses, size=number_of_synapses, replace=False
        )
        for src, dest in used_syanpses:
            polarity = self._rng.choice([-1, 1, 1])
            self.add_synapse(src, dest, polarity)

    def build(self, input_shape):
        super().build(input_shape)
        number_of_sensory_synapses = int(
            np.round(self.input_dim * self.units * (1 - self.sparsity_level))
        )
        all_sensory_synapses = []
        for src in range(self.input_dim):
            for dest in range(self.units):
                all_sensory_synapses.append((src, dest))

        used_sensory_syanpses = self._rng.choice(
            all_sensory_synapses, size=number_of_sensory_synapses, replace=False
        )
        for src, dest in used_sensory_syanpses:
            polarity = self._rng.choice([-1, 1, 1])
            self.add_sensory_synapse(src, dest, polarity)
            polarity = self._rng.choice([-1, 1, 1])
            self.add_sensory_synapse(src, dest, polarity)


class NCP(Wiring):
    def __init__(
        self,
        inter_neurons,
        command_neurons,
        motor_neurons,
        sensory_fanout,
        inter_fanout,
        recurrent_command_synapses,
        motor_fanin,
        seed=22222,
    ):

        super(NCP, self).__init__(inter_neurons + command_neurons + motor_neurons)
        self.set_output_dim(motor_neurons)
        self._rng = np.random.RandomState(seed)
        self._num_inter_neurons = inter_neurons
        self._num_command_neurons = command_neurons
        self._num_motor_neurons = motor_neurons
        self._sensory_fanout = sensory_fanout
        self._inter_fanout = inter_fanout
        self._recurrent_command_synapses = recurrent_command_synapses
        self._motor_fanin = motor_fanin

        # Neuron IDs: [0..motor ... command ... inter]
        self._motor_neurons = list(range(0, self._num_motor_neurons))
        self._command_neurons = list(
            range(
                self._num_motor_neurons,
                self._num_motor_neurons + self._num_command_neurons,
            )
        )
        self._inter_neurons = list(
            range(
                self._num_motor_neurons + self._num_command_neurons,
                self._num_motor_neurons
                + self._num_command_neurons
                + self._num_inter_neurons,
            )
        )

        if self._motor_fanin > self._num_command_neurons:
            raise ValueError(
                "Error: Motor fanin parameter is {} but there are only {} command neurons".format(
                    self._motor_fanin, self._num_command_neurons
                )
            )
        if self._sensory_fanout > self._num_inter_neurons:
            raise ValueError(
                "Error: Sensory fanout parameter is {} but there are only {} inter neurons".format(
                    self._sensory_fanout, self._num_inter_neurons
                )
            )
        if self._inter_fanout > self._num_command_neurons:
            raise ValueError(
                "Error:: Inter fanout parameter is {} but there are only {} command neurons".format(
                    self._inter_fanout, self._num_command_neurons
                )
            )

    def get_type_of_neuron(self, neuron_id):
        if neuron_id < self._num_motor_neurons:
            return "motor"
        if neuron_id < self._num_motor_neurons + self._num_command_neurons:
            return "command"
        return "inter"

    def _build_sensory_to_inter_layer(self):
        unreachable_inter_neurons = [l for l in self._inter_neurons]
        # Randomly connects each sensory neuron to exactly _sensory_fanout number of interneurons
        for src in self._sensory_neurons:
            for dest in self._rng.choice(
                self._inter_neurons, size=self._sensory_fanout, replace=False
            ):
                if dest in unreachable_inter_neurons:
                    unreachable_inter_neurons.remove(dest)
                polarity = self._rng.choice([-1, 1])
                self.add_sensory_synapse(src, dest, polarity)

        # If it happens that some interneurons are not connected, connect them now
        mean_inter_neuron_fanin = int(
            self._num_sensory_neurons * self._sensory_fanout / self._num_inter_neurons
        )
        # Connect "forgotten" inter neuron by at least 1 and at most all sensory neuron
        mean_inter_neuron_fanin = np.clip(
            mean_inter_neuron_fanin, 1, self._num_sensory_neurons
        )
        for dest in unreachable_inter_neurons:
            for src in self._rng.choice(
                self._sensory_neurons, size=mean_inter_neuron_fanin, replace=False
            ):
                polarity = self._rng.choice([-1, 1])
                self.add_sensory_synapse(src, dest, polarity)

    def _build_inter_to_command_layer(self):
        # Randomly connect interneurons to command neurons
        unreachable_command_neurons = [l for l in self._command_neurons]
        for src in self._inter_neurons:
            for dest in self._rng.choice(
                self._command_neurons, size=self._inter_fanout, replace=False
            ):
                if dest in unreachable_command_neurons:
                    unreachable_command_neurons.remove(dest)
                polarity = self._rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

        # If it happens that some command neurons are not connected, connect them now
        mean_command_neurons_fanin = int(
            self._num_inter_neurons * self._inter_fanout / self._num_command_neurons
        )
        # Connect "forgotten" command neuron by at least 1 and at most all inter neuron
        mean_command_neurons_fanin = np.clip(
            mean_command_neurons_fanin, 1, self._num_command_neurons
        )
        for dest in unreachable_command_neurons:
            for src in self._rng.choice(
                self._inter_neurons, size=mean_command_neurons_fanin, replace=False
            ):
                polarity = self._rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

    def _build_recurrent_command_layer(self):
        # Add recurrency in command neurons
        for i in range(self._recurrent_command_synapses):
            src = self._rng.choice(self._command_neurons)
            dest = self._rng.choice(self._command_neurons)
            polarity = self._rng.choice([-1, 1])
            self.add_synapse(src, dest, polarity)

    def _build_command__to_motor_layer(self):
        # Randomly connect command neurons to motor neurons
        unreachable_command_neurons = [l for l in self._command_neurons]
        for dest in self._motor_neurons:
            for src in self._rng.choice(
                self._command_neurons, size=self._motor_fanin, replace=False
            ):
                if src in unreachable_command_neurons:
                    unreachable_command_neurons.remove(src)
                polarity = self._rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

        # If it happens that some commandneurons are not connected, connect them now
        mean_command_fanout = int(
            self._num_motor_neurons * self._motor_fanin / self._num_command_neurons
        )
        # Connect "forgotten" command neuron to at least 1 and at most all motor neuron
        mean_command_fanout = np.clip(mean_command_fanout, 1, self._num_motor_neurons)
        for src in unreachable_command_neurons:
            for dest in self._rng.choice(
                self._motor_neurons, size=mean_command_fanout, replace=False
            ):
                polarity = self._rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

    def build(self, input_shape):
        super().build(input_shape)
        self._num_sensory_neurons = self.input_dim
        self._sensory_neurons = list(range(0, self._num_sensory_neurons))

        self._build_sensory_to_inter_layer()
        self._build_inter_to_command_layer()
        self._build_recurrent_command_layer()
        self._build_command__to_motor_layer()

