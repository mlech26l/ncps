import paddle
import numpy as np
import paddle.nn as nn
import kerasncp as kncp
from paddle.optimizer import Adam
from kerasncp.paddle import LTCCell
from paddle.io import DataLoader, TensorDataset


class RNNSequence(nn.Layer):
    def __init__(
        self,
        rnn_cell,
    ):
        super(RNNSequence, self).__init__()
        self.rnn_cell = rnn_cell

    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        hidden_state = paddle.zeros((batch_size, self.rnn_cell.state_size))
        outputs = []
        for t in range(seq_len):
            inputs = x[:, t]
            new_output, hidden_state = self.rnn_cell.forward(
                inputs, hidden_state)
            outputs.append(new_output)
        outputs = paddle.stack(outputs, axis=1)  # return entire sequence
        return outputs


class SequenceLearner(paddle.Model):
    def train_batch(self, inputs, labels=None, update=True):
        x, y = inputs[0], labels[0]
        y_hat = self.network.forward(x)
        y_hat = y_hat.reshape(y.shape)
        loss = self._loss(y_hat, y)
        loss.backward()
        if update:
            self._optimizer.step()
            self._optimizer.clear_grad()
            self.network.rnn_cell.apply_weight_constraints()
        return [loss.numpy()]

    def eval_batch(self, inputs, labels=None):
        x, y = inputs[0], labels[0]
        y_hat = self.network.forward(x)
        y_hat = y_hat.reshape(y.shape)
        loss = self._loss(y_hat, y)
        return [loss.numpy()]

    def predict_batch(self, inputs):
        x = inputs[0]
        y_hat = self.network.forward(x)
        return [x.numpy(), y_hat.numpy()]


if __name__ == '__main__':
    in_features = 2
    out_features = 1
    N = 48  # Length of the time-series
    # Input feature is a sine and a cosine wave
    data_x = np.stack(
        [np.sin(np.linspace(0, 3 * np.pi, N)), np.cos(np.linspace(0, 3 * np.pi, N))], axis=1
    )
    data_x = np.expand_dims(data_x, axis=0).astype(
        np.float32)  # Add batch dimension
    # Target output is a sine with double the frequency of the input signal
    data_y = np.sin(np.linspace(0, 6 * np.pi, N)
                    ).reshape([1, N, 1]).astype(np.float32)
    data_x = paddle.to_tensor(data_x)
    data_y = paddle.to_tensor(data_y)
    print("data_y.shape: ", str(data_y.shape))

    wiring = kncp.wirings.FullyConnected(
        8, out_features)  # 16 units, 8 motor neurons
    ltc_cell = LTCCell(wiring, in_features)
    dataloader = DataLoader(TensorDataset(
        [data_x, data_y]), batch_size=1, shuffle=True, num_workers=4)

    ltc_sequence = RNNSequence(ltc_cell)
    learn = SequenceLearner(ltc_sequence)
    opt = Adam(learning_rate=0.01, parameters=ltc_sequence.parameters())
    loss = nn.MSELoss()
    learn.prepare(opt, loss)
    learn.fit(dataloader, epochs=400, verbose=1)
    results = learn.evaluate(dataloader)
