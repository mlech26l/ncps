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


import numpy as np


class Wiring:
    def __init__(self, units):
        self.units = units
        self.adjacency_matrix = np.zeros([units, units], dtype=np.int32)
        self.sensory_adjacency_matrix = None
        self.input_dim = None
        self.output_dim = None

    @property
    def num_layers(self):
        return 1

    def get_neurons_of_layer(self, layer_id):
        return list(range(self.units))

    def is_built(self):
        return self.input_dim is not None

    def build(self, input_dim):
        if not self.input_dim is None and self.input_dim != input_dim:
            raise ValueError(
                "Conflicting input dimensions provided. set_input_dim() was called with {} but actual input has dimension {}".format(
                    self.input_dim, input_dim
                )
            )
        if self.input_dim is None:
            self.set_input_dim(input_dim)

    def erev_initializer(self, shape=None, dtype=None):
        return np.copy(self.adjacency_matrix)

    def sensory_erev_initializer(self, shape=None, dtype=None):
        return np.copy(self.sensory_adjacency_matrix)

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

    def get_config(self):
        return {
            "adjacency_matrix": self.adjacency_matrix,
            "sensory_adjacency_matrix": self.sensory_adjacency_matrix,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "units": self.units,
        }

    @classmethod
    def from_config(cls, config):
        # There might be a cleaner solution but it will work
        wiring = Wiring(config["units"])
        wiring.adjacency_matrix = config["adjacency_matrix"]
        wiring.sensory_adjacency_matrix = config["sensory_adjacency_matrix"]
        wiring.input_dim = config["input_dim"]
        wiring.output_dim = config["output_dim"]

        return wiring

    def get_graph(self, include_sensory_neurons=True):
        """
        Returns a networkx.DiGraph object of the wiring diagram
        :param include_sensory_neurons: Whether to include the sensory neurons as nodes in the graph
        """
        if not self.is_built():
            raise ValueError(
                "Wiring is not built yet.\n"
                "This is probably because the input shape is not known yet.\n"
                "Consider calling the model.build(...) method using the shape of the inputs."
            )
        # Only import networkx package if we really need it
        import networkx as nx

        DG = nx.DiGraph()
        for i in range(self.units):
            neuron_type = self.get_type_of_neuron(i)
            DG.add_node("neuron_{:d}".format(i), neuron_type=neuron_type)
        for i in range(self.input_dim):
            DG.add_node("sensory_{:d}".format(i), neuron_type="sensory")

        erev = self.adjacency_matrix
        sensory_erev = self.sensory_adjacency_matrix

        for src in range(self.input_dim):
            for dest in range(self.units):
                if self.sensory_adjacency_matrix[src, dest] != 0:
                    polarity = (
                        "excitatory" if sensory_erev[src, dest] >= 0.0 else "inhibitory"
                    )
                    DG.add_edge(
                        "sensory_{:d}".format(src),
                        "neuron_{:d}".format(dest),
                        polarity=polarity,
                    )

        for src in range(self.units):
            for dest in range(self.units):
                if self.adjacency_matrix[src, dest] != 0:
                    polarity = "excitatory" if erev[src, dest] >= 0.0 else "inhibitory"
                    DG.add_edge(
                        "neuron_{:d}".format(src),
                        "neuron_{:d}".format(dest),
                        polarity=polarity,
                    )
        return DG

    @property
    def synapse_count(self):
        """Counts the number of synapses between internal neurons of the model"""
        return np.sum(np.abs(self.adjacency_matrix))

    @property
    def sensory_synapse_count(self):
        """Counts the number of synapses from the inputs (sensory neurons) to the internal neurons of the model"""
        return np.sum(np.abs(self.sensory_adjacency_matrix))

    def draw_graph(
        self,
        layout="shell",
        neuron_colors=None,
        synapse_colors=None,
        draw_labels=False,
    ):
        """Draws a matplotlib graph of the wiring structure
        Examples::

            >>> import matplotlib.pyplot as plt
            >>> plt.figure(figsize=(6, 4))
            >>> legend_handles = wiring.draw_graph(draw_labels=True)
            >>> plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
            >>> plt.tight_layout()
            >>> plt.show()

        :param layout:
        :param neuron_colors:
        :param synapse_colors:
        :param draw_labels:
        :return:
        """

        # May switch to Cytoscape once support in Google Colab is available
        # https://stackoverflow.com/questions/62421021/how-do-i-install-cytoscape-on-google-colab
        import networkx as nx
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        if isinstance(synapse_colors, str):
            synapse_colors = {
                "excitatory": synapse_colors,
                "inhibitory": synapse_colors,
            }
        elif synapse_colors is None:
            synapse_colors = {"excitatory": "tab:green", "inhibitory": "tab:red"}

        default_colors = {
            "inter": "tab:blue",
            "motor": "tab:orange",
            "sensory": "tab:olive",
        }
        if neuron_colors is None:
            neuron_colors = {}
        # Merge default with user provided color dict
        for k, v in default_colors.items():
            if not k in neuron_colors.keys():
                neuron_colors[k] = v

        legend_patches = []
        for k, v in neuron_colors.items():
            label = "{}{} neurons".format(k[0].upper(), k[1:])
            color = v
            legend_patches.append(mpatches.Patch(color=color, label=label))

        G = self.get_graph()
        layouts = {
            "kamada": nx.kamada_kawai_layout,
            "circular": nx.circular_layout,
            "random": nx.random_layout,
            "shell": nx.shell_layout,
            "spring": nx.spring_layout,
            "spectral": nx.spectral_layout,
            "spiral": nx.spiral_layout,
        }
        if not layout in layouts.keys():
            raise ValueError(
                "Unknown layer '{}', use one of '{}'".format(
                    layout, str(layouts.keys())
                )
            )
        pos = layouts[layout](G)

        # Draw neurons
        for i in range(self.units):
            node_name = "neuron_{:d}".format(i)
            neuron_type = G.nodes[node_name]["neuron_type"]
            neuron_color = "tab:blue"
            if neuron_type in neuron_colors.keys():
                neuron_color = neuron_colors[neuron_type]
            nx.draw_networkx_nodes(G, pos, [node_name], node_color=neuron_color)

        # Draw sensory neurons
        for i in range(self.input_dim):
            node_name = "sensory_{:d}".format(i)
            neuron_color = "blue"
            if "sensory" in neuron_colors.keys():
                neuron_color = neuron_colors["sensory"]
            nx.draw_networkx_nodes(G, pos, [node_name], node_color=neuron_color)

        # Optional: draw labels
        if draw_labels:
            nx.draw_networkx_labels(G, pos)

        # Draw edges
        for node1, node2, data in G.edges(data=True):
            polarity = data["polarity"]
            edge_color = synapse_colors[polarity]
            nx.draw_networkx_edges(G, pos, [(node1, node2)], edge_color=edge_color)

        return legend_patches


class FullyConnected(Wiring):
    def __init__(
        self, units, output_dim=None, erev_init_seed=1111, self_connections=True
    ):
        super(FullyConnected, self).__init__(units)
        if output_dim is None:
            output_dim = units
        self.self_connections = self_connections
        self.set_output_dim(output_dim)
        self._rng = np.random.default_rng(erev_init_seed)
        for src in range(self.units):
            for dest in range(self.units):
                if src == dest and not self_connections:
                    continue
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

        used_synapses = self._rng.choice(
            all_synapses, size=number_of_synapses, replace=False
        )
        for src, dest in used_synapses:
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

        used_sensory_synapses = self._rng.choice(
            all_sensory_synapses, size=number_of_sensory_synapses, replace=False
        )
        for src, dest in used_sensory_synapses:
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
        """
        Creates a Neural Circuit Policies wiring.
        The total number of neurons (= state size of the RNN) is given by the sum of inter, command, and motor neurons.
        For an easier way to generate a NCP wiring see the ``AutoNCP`` wiring class.

        :param inter_neurons: The number of inter neurons (layer 2)
        :param command_neurons: The number of command neurons (layer 3)
        :param motor_neurons: The number of motor neurons (layer 4 = number of outputs)
        :param sensory_fanout: The average number of outgoing synapses from the sensory to the inter neurons
        :param inter_fanout: The average number of outgoing synapses from the inter to the command neurons
        :param recurrent_command_synapses: The average number of recurrent connections in the command neuron layer
        :param motor_fanin: The average number of incoming synapses of the motor neurons from the command neurons
        :param seed: The random seed used to generate the wiring
        """

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

    @property
    def num_layers(self):
        return 3

    def get_neurons_of_layer(self, layer_id):
        if layer_id == 0:
            return self._inter_neurons
        elif layer_id == 1:
            return self._command_neurons
        elif layer_id == 2:
            return self._motor_neurons
        raise ValueError("Unknown layer {}".format(layer_id))

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


class AutoNCP(NCP):
    def __init__(
        self,
        units,
        output_size,
        sparsity_level=0.5,
        seed=22222,
    ):
        """Instantiate an NCP wiring with only needing to specify the number of units and the number of outputs

        :param units: The total number of neurons
        :param output_size: The number of motor neurons (=output size). This value must be less than units-2 (typically good choices are 0.3 times the total number of units)
        :param sparsity_level: A hyperparameter between 0.0 (very dense) and 0.9 (very sparse) NCP.
        :param seed: Random seed for generating the wiring
        """
        if output_size >= units - 2:
            raise ValueError(
                f"Output size must be less than the number of units-2 (given {units} units, {output_size} output size)"
            )
        if sparsity_level < 0.1 or sparsity_level > 1.0:
            raise ValueError(
                f"Sparsity level must be between 0.0 and 0.9 (given {sparsity_level})"
            )
        density_level = 1.0 - sparsity_level
        inter_and_command_neurons = units - output_size
        command_neurons = max(int(0.4 * inter_and_command_neurons), 1)
        inter_neurons = inter_and_command_neurons - command_neurons

        sensory_fanout = max(int(inter_neurons * density_level), 1)
        inter_fanout = max(int(command_neurons * density_level), 1)
        recurrent_command_synapses = max(int(command_neurons * density_level * 2), 1)
        motor_fanin = max(int(command_neurons * density_level), 1)
        super(AutoNCP, self).__init__(
            inter_neurons,
            command_neurons,
            output_size,
            sensory_fanout,
            inter_fanout,
            recurrent_command_synapses,
            motor_fanin,
            seed=seed,
        )