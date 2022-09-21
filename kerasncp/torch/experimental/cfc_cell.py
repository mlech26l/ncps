import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_map = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.recurrent_map = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for w in self.input_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.xavier_uniform_(w)
        for w in self.recurrent_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.orthogonal_(w)

    def forward(self, inputs, states):
        output_state, cell_state = states
        z = self.input_map(inputs) + self.recurrent_map(output_state)
        i, ig, fg, og = z.chunk(4, 1)

        input_activation = self.tanh(i)
        input_gate = self.sigmoid(ig)
        forget_gate = self.sigmoid(fg + 1.0)
        output_gate = self.sigmoid(og)

        new_cell = cell_state * forget_gate + input_activation * input_gate
        output_state = self.tanh(new_cell) * output_gate

        return output_state, new_cell


class LeCun(nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


class CfCCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        mode="default",
        backbone_activation="lecun",
        backbone_units=128,
        backbone_layers=1,
        backbone_dr=0.0,
        sparsity_mask=None,
    ):
        super(CfCCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        allowed_modes = ["default", "pure", "no_gate"]
        if mode not in allowed_modes:
            raise ValueError(
                f"Unknown mode '{mode}', valid options are {str(allowed_modes)}"
            )
        self.sparsity_mask = (
            None
            if sparsity_mask is None
            else torch.nn.Parameter(
                data=torch.from_numpy(np.abs(sparsity_mask.T).astype(np.float32)),
                requires_grad=False,
            )
        )

        self.mode = mode

        if backbone_activation == "silu":
            backbone_activation = nn.SiLU
        elif backbone_activation == "relu":
            backbone_activation = nn.ReLU
        elif backbone_activation == "tanh":
            backbone_activation = nn.Tanh
        elif backbone_activation == "gelu":
            backbone_activation = nn.GELU
        elif backbone_activation == "lecun":
            backbone_activation = LeCun
        else:
            raise ValueError(f"Unknown activation {backbone_activation}")
        layer_list = [
            nn.Linear(input_size + hidden_size, backbone_units),
            backbone_activation(),
        ]
        for i in range(1, backbone_layers):
            layer_list.append(nn.Linear(backbone_units, backbone_units))
            layer_list.append(backbone_activation())
            if backbone_dr > 0.0:
                layer_list.append(torch.nn.Dropout(backbone_dr))
        self.backbone = nn.Sequential(*layer_list)
        self.backbone_layers = backbone_layers
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        cat_shape = int(
            self.hidden_size + input_size if backbone_layers == 0 else backbone_units
        )

        self.ff1 = nn.Linear(cat_shape, hidden_size)
        if self.mode == "pure":
            self.w_tau = torch.nn.Parameter(
                data=torch.zeros(1, self.hidden_size), requires_grad=True
            )
            self.A = torch.nn.Parameter(
                data=torch.ones(1, self.hidden_size), requires_grad=True
            )
        else:
            self.ff2 = nn.Linear(cat_shape, hidden_size)
            self.time_a = nn.Linear(cat_shape, hidden_size)
            self.time_b = nn.Linear(cat_shape, hidden_size)
        self.init_weights()

    def init_weights(self):
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                torch.nn.init.xavier_uniform_(w)

    def forward(self, input, hx, ts):

        batch_size = input.size(0)
        x = torch.cat([input, hx], 1)
        if self.backbone_layers > 0:
            x = self.backbone(x)
        if self.sparsity_mask is not None:
            ff1 = F.linear(x, self.ff1.weight * self.sparsity_mask, self.ff1.bias)
        else:
            ff1 = self.ff1(x)
        if self.mode == "pure":
            # Solution
            new_hidden = (
                -self.A
                * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(ff1)))
                * ff1
                + self.A
            )
        else:
            # Cfc
            if self.sparsity_mask is not None:
                ff2 = F.linear(x, self.ff2.weight * self.sparsity_mask, self.ff2.bias)
            else:
                ff2 = self.ff2(x)
            ff1 = self.tanh(ff1)
            ff2 = self.tanh(ff2)
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            t_interp = self.sigmoid(t_a * ts + t_b)
            if self.mode == "no_gate":
                new_hidden = ff1 + t_interp * ff2
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
        return new_hidden


class CfC(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_size,
        out_features=None,
        return_sequences=True,
        use_mm_rnn=False,
        mode="default",
        backbone_activation="lecun",
        backbone_units=128,
        backbone_layers=1,
        backbone_dr=0.0,
    ):
        super(CfC, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_features = out_features
        self.return_sequences = return_sequences

        self.rnn_cell = CfCCell(
            in_features,
            hidden_size,
            mode,
            backbone_activation,
            backbone_units,
            backbone_layers,
            backbone_dr,
        )
        self.use_mixed = use_mm_rnn
        if self.use_mixed:
            self.lstm = LSTMCell(in_features, hidden_size)

        if out_features is None:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(self.hidden_size, self.out_features)

    def forward(self, x, timespans=None):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        h_state = torch.zeros((batch_size, self.hidden_size), device=device)
        if self.use_mixed:
            c_state = torch.zeros((batch_size, self.hidden_size), device=device)
        output_sequence = []
        for t in range(seq_len):
            inputs = x[:, t]
            ts = 1.0 if timespans is None else timespans[:, t].squeeze()
            if self.use_mixed:
                h_state, c_state = self.lstm(inputs, (h_state, c_state))
            h_state = self.rnn_cell.forward(inputs, h_state, ts)
            if self.return_sequences:
                output_sequence.append(self.fc(h_state))

        if self.return_sequences:
            readout = torch.stack(output_sequence, dim=1)
        else:
            readout = self.fc(h_state)
        return readout


class WiredCfC(nn.Module):
    def __init__(
        self,
        wiring,
        in_features=None,
        return_sequences=True,
        use_mm_rnn=False,
        mode="default",
    ):
        super(WiredCfC, self).__init__()

        if in_features is not None:
            wiring.build(in_features)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown number of input features. Please pass the parameter 'in_features' or call the 'wiring.build()'."
            )
        self._wiring = wiring
        self.return_sequences = return_sequences

        self._layers = []
        in_features = wiring.input_dim
        for l in range(wiring.num_layers):
            hidden_units = self._wiring.get_neurons_of_layer(l)
            if l == 0:
                input_sparsity = self._wiring.sensory_adjacency_matrix[:, hidden_units]
            else:
                prev_layer_neurons = self._wiring.get_neurons_of_layer(l - 1)
                input_sparsity = self._wiring.adjacency_matrix[:, hidden_units]
                input_sparsity = input_sparsity[prev_layer_neurons, :]
            input_sparsity = np.concatenate(
                [
                    input_sparsity,
                    np.ones((len(hidden_units), len(hidden_units))),
                ],
                axis=0,
            )

            # Hack: nn.Module registers child params in set_attribute
            rnn_cell = CfCCell(
                in_features,
                len(hidden_units),
                mode,
                backbone_activation="lecun",
                backbone_units=0,
                backbone_layers=0,
                backbone_dr=0.0,
                sparsity_mask=input_sparsity,
            )
            self.register_module(f"layer_{l}", rnn_cell)
            self._layers.append(rnn_cell)
            in_features = len(hidden_units)
        self.use_mixed = use_mm_rnn
        if self.use_mixed:
            self.lstm = LSTMCell(self.sensory_size, self.state_size)

    @property
    def state_size(self):
        return self._wiring.units

    @property
    def layer_sizes(self):
        return [
            len(self._wiring.get_neurons_of_layer(i))
            for i in range(self._wiring.num_layers)
        ]

    @property
    def num_layers(self):
        return self._wiring.num_layers

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

    def forward(self, x, timespans=None):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        h_state = torch.zeros((batch_size, self.state_size), device=device)
        h_state = torch.split(h_state, self.layer_sizes, dim=1)
        if self.use_mixed:
            c_state = torch.zeros((batch_size, self.state_size), device=device)
        output_sequence = []
        for t in range(seq_len):
            inputs = x[:, t]
            ts = 1.0 if timespans is None else timespans[:, t].squeeze()
            if self.use_mixed:
                h_state = torch.concat(h_state, dim=1)
                h_state, c_state = self.lstm(x[:, t], (h_state, c_state))
                h_state = torch.split(h_state, self.layer_sizes, dim=1)

            new_h_state = []
            for i in range(self.num_layers):
                h = self._layers[i].forward(inputs, h_state[i], ts)
                inputs = h
                new_h_state.append(h)

            h_state = new_h_state
            if self.return_sequences:
                output_sequence.append(h_state[-1])  # motor neurons

        if self.return_sequences:
            readout = torch.stack(output_sequence, dim=1)
        else:
            readout = h_state[-1]
        return readout