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

from kerasncp import wirings
import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Custom", name="LTCCell")
class LTCCell(tf.keras.layers.AbstractRNNCell):
    def __init__(
        self,
        wiring,
        input_mapping="affine",
        output_mapping="affine",
        ode_unfolds=6,
        epsilon=1e-8,
        initialization_ranges=None,
        **kwargs
    ):

        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        if not initialization_ranges is None:
            for k, v in initialization_ranges.items():
                if k not in self._init_ranges.keys():
                    raise ValueError(
                        "Unknown parameter '{}' in initialization range dictionary! (Expected only {})".format(
                            k, str(list(self._init_range.keys()))
                        )
                    )
                if k in ["gleak", "cm", "w", "sensory_w"] and v[0] < 0:
                    raise ValueError(
                        "Initialization range of parameter '{}' must be non-negative!".format(
                            k
                        )
                    )
                if v[0] > v[1]:
                    raise ValueError(
                        "Initialization range of parameter '{}' is not a valid range".format(
                            k
                        )
                    )
                self._init_ranges[k] = v

        self._wiring = wiring
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        super(LTCCell, self).__init__(name="ltc_cell")

    @property
    def state_size(self):
        return self._wiring.units

    @property
    def sensory_size(self):
        return self._wiring.input_dim

    @property
    def motor_size(self):
        return self._wiring.output_dim

    @property
    def output_size(self):
        return self.motor_size

    @property
    def synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    @property
    def sensory_synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    def _get_initializer(self, param_name):
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return tf.keras.initializers.Constant(minval)
        else:
            return tf.keras.initializers.RandomUniform(minval, maxval)

    def build(self, input_shape):

        # Check if input_shape is nested tuple/list
        if isinstance(input_shape[0], (tuple, list)):
            input_shape = input_shape[0]

        self._wiring.build(input_shape)

        self._params = {}
        self._params["gleak"] = self.add_weight(
            name="gleak",
            shape=(self.state_size,),
            dtype=tf.float32,
            constraint=tf.keras.constraints.NonNeg(),
            initializer=self._get_initializer("gleak"),
        )
        self._params["vleak"] = self.add_weight(
            name="vleak",
            shape=(self.state_size,),
            dtype=tf.float32,
            initializer=self._get_initializer("vleak"),
        )
        self._params["cm"] = self.add_weight(
            name="cm",
            shape=(self.state_size,),
            dtype=tf.float32,
            constraint=tf.keras.constraints.NonNeg(),
            initializer=self._get_initializer("cm"),
        )
        self._params["sigma"] = self.add_weight(
            name="sigma",
            shape=(self.state_size, self.state_size),
            dtype=tf.float32,
            initializer=self._get_initializer("sigma"),
        )
        self._params["mu"] = self.add_weight(
            name="mu",
            shape=(self.state_size, self.state_size),
            dtype=tf.float32,
            initializer=self._get_initializer("mu"),
        )
        self._params["w"] = self.add_weight(
            name="w",
            shape=(self.state_size, self.state_size),
            dtype=tf.float32,
            constraint=tf.keras.constraints.NonNeg(),
            initializer=self._get_initializer("w"),
        )
        self._params["erev"] = self.add_weight(
            name="erev",
            shape=(self.state_size, self.state_size),
            dtype=tf.float32,
            initializer=self._wiring.erev_initializer,
        )

        self._params["sensory_sigma"] = self.add_weight(
            name="sensory_sigma",
            shape=(self.sensory_size, self.state_size),
            dtype=tf.float32,
            initializer=self._get_initializer("sensory_sigma"),
        )
        self._params["sensory_mu"] = self.add_weight(
            name="sensory_mu",
            shape=(self.sensory_size, self.state_size),
            dtype=tf.float32,
            initializer=self._get_initializer("sensory_mu"),
        )
        self._params["sensory_w"] = self.add_weight(
            name="sensory_w",
            shape=(self.sensory_size, self.state_size),
            dtype=tf.float32,
            constraint=tf.keras.constraints.NonNeg(),
            initializer=self._get_initializer("sensory_w"),
        )
        self._params["sensory_erev"] = self.add_weight(
            name="sensory_erev",
            shape=(self.sensory_size, self.state_size),
            dtype=tf.float32,
            initializer=self._wiring.sensory_erev_initializer,
        )

        self._params["sparsity_mask"] = tf.constant(
            np.abs(self._wiring.adjacency_matrix), dtype=tf.float32
        )
        self._params["sensory_sparsity_mask"] = tf.constant(
            np.abs(self._wiring.sensory_adjacency_matrix), dtype=tf.float32
        )

        if self._input_mapping in ["affine", "linear"]:
            self._params["input_w"] = self.add_weight(
                name="input_w",
                shape=(self.sensory_size,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(1),
            )
        if self._input_mapping == "affine":
            self._params["input_b"] = self.add_weight(
                name="input_b",
                shape=(self.sensory_size,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(0),
            )

        if self._output_mapping in ["affine", "linear"]:
            self._params["output_w"] = self.add_weight(
                name="output_w",
                shape=(self.motor_size,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(1),
            )
        if self._output_mapping == "affine":
            self._params["output_b"] = self.add_weight(
                name="output_b",
                shape=(self.motor_size,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(0),
            )
        self.built = True

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = tf.expand_dims(v_pre, axis=-1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return tf.nn.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self._params["sensory_w"] * self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )
        sensory_w_activation *= self._params["sensory_sparsity_mask"]

        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]

        # Reduce over dimension 1 (=source sensory neurons)
        w_numerator_sensory = tf.reduce_sum(sensory_rev_activation, axis=1)
        w_denominator_sensory = tf.reduce_sum(sensory_w_activation, axis=1)

        # cm/t is loop invariant
        cm_t = self._params["cm"] / tf.cast(
            elapsed_time / self._ode_unfolds, dtype=tf.float32
        )

        # Unfold the multiply ODE multiple times into one RNN step
        for t in range(self._ode_unfolds):
            w_activation = self._params["w"] * self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )

            w_activation *= self._params["sparsity_mask"]

            rev_activation = w_activation * self._params["erev"]

            # Reduce over dimension 1 (=source neurons)
            w_numerator = tf.reduce_sum(rev_activation, axis=1) + w_numerator_sensory
            w_denominator = tf.reduce_sum(w_activation, axis=1) + w_denominator_sensory

            numerator = (
                cm_t * v_pre
                + self._params["gleak"] * self._params["vleak"]
                + w_numerator
            )
            denominator = cm_t + self._params["gleak"] + w_denominator

            # Avoid dividing by 0
            v_pre = numerator / (denominator + self._epsilon)

        return v_pre

    def _map_inputs(self, inputs):
        if self._input_mapping in ["affine", "linear"]:
            inputs = inputs * self._params["input_w"]
        if self._input_mapping == "affine":
            inputs = inputs + self._params["input_b"]
        return inputs

    def _map_outputs(self, state):
        output = state
        if self.motor_size < self.state_size:
            output = output[:, 0 : self.motor_size]

        if self._output_mapping in ["affine", "linear"]:
            output = output * self._params["output_w"]
        if self._output_mapping == "affine":
            output = output + self._params["output_b"]
        return output

    def call(self, inputs, states):
        if isinstance(inputs, (tuple, list)):
            # Irregularly sampled mode
            inputs, elapsed_time = inputs
        else:
            # Regularly sampled mode (elapsed time = 1 second)
            elapsed_time = 1.0
        inputs = self._map_inputs(inputs)

        next_state = self._ode_solver(inputs, states[0], elapsed_time)

        outputs = self._map_outputs(next_state)

        return outputs, [next_state]

    def get_config(self):
        seralized = self._wiring.get_config()
        seralized["input_mapping"] = self._input_mapping
        seralized["output_mapping"] = self._output_mapping
        seralized["ode_unfolds"] = self._ode_unfolds
        seralized["epsilon"] = self._epsilon
        return seralized

    @classmethod
    def from_config(cls, config):
        wiring = wirings.Wiring.from_config(config)
        return cls(wiring=wiring, **config)

    def get_graph(self, include_sensory_neurons=True):
        if not self.built:
            raise ValueError(
                "LTCCell layer is not built yet.\n"
                "This is probably because the input shape is not known yet.\n"
                "Consider calling the model.build(...) method using the shape of the inputs."
            )
        # Only import networkx package if we really need it
        import networkx as nx

        DG = nx.DiGraph()
        for i in range(self.state_size):
            neuron_type = self._wiring.get_type_of_neuron(i)
            DG.add_node("neuron_{:d}".format(i), neuron_type=neuron_type)
        for i in range(self.sensory_size):
            DG.add_node("sensory_{:d}".format(i), neuron_type="sensory")

        erev = self._params["erev"].numpy()
        sensory_erev = self._params["sensory_erev"].numpy()

        for src in range(self.sensory_size):
            for dest in range(self.state_size):
                if self._wiring.sensory_adjacency_matrix[src, dest] != 0:
                    polarity = (
                        "excitatory" if sensory_erev[src, dest] >= 0.0 else "inhibitory"
                    )
                    DG.add_edge(
                        "sensory_{:d}".format(src),
                        "neuron_{:d}".format(dest),
                        polarity=polarity,
                    )

        for src in range(self.state_size):
            for dest in range(self.state_size):
                if self._wiring.adjacency_matrix[src, dest] != 0:
                    polarity = "excitatory" if erev[src, dest] >= 0.0 else "inhibitory"
                    DG.add_edge(
                        "neuron_{:d}".format(src),
                        "neuron_{:d}".format(dest),
                        polarity=polarity,
                    )
        return DG

    def draw_graph(
        self,
        layout="shell",
        neuron_colors=None,
        synapse_colors=None,
        draw_labels=False,
    ):
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
        for i in range(self.state_size):
            node_name = "neuron_{:d}".format(i)
            neuron_type = G.nodes[node_name]["neuron_type"]
            neuron_color = "tab:blue"
            if neuron_type in neuron_colors.keys():
                neuron_color = neuron_colors[neuron_type]
            nx.draw_networkx_nodes(G, pos, [node_name], node_color=neuron_color)

        # Draw sensory neurons
        for i in range(self.sensory_size):
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
