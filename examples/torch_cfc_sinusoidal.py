# Copyright (2017-2021)
# The Wormnet project
# Mathias Lechner (mlechner@ist.ac.at)
import numpy as np
import torch.nn as nn
import kerasncp as kncp
import pytorch_lightning as pl
import torch
import torch.utils.data as data

from kerasncp.torch.experimental import CfC, WiredCfC


# LightningModule for training a RNNSequence module
class SequenceLearner(pl.LightningModule):
    def __init__(self, model, lr=0.005):
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model.forward(x)
        # y_hat = y_hat.view_as(y)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = nn.MSELoss()(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


in_features = 2
out_features = 1
N = 48  # Length of the time-series
# Input feature is a sine and a cosine wave
data_x = np.stack(
    [np.sin(np.linspace(0, 3 * np.pi, N)), np.cos(np.linspace(0, 3 * np.pi, N))], axis=1
)
data_x = np.expand_dims(data_x, axis=0).astype(np.float32)  # Add batch dimension
# Target output is a sine with double the frequency of the input signal
data_y = np.sin(np.linspace(0, 6 * np.pi, N)).reshape([1, N, 1]).astype(np.float32)
data_x = torch.Tensor(data_x)
data_y = torch.Tensor(data_y)
print("data_y.shape: ", str(data_y.shape))

for model in [
    CfC(in_features=in_features, hidden_size=32, out_features=out_features),
    WiredCfC(
        in_features=in_features, wiring=kncp.wirings.FullyConnected(8, out_features)
    ),
    WiredCfC(
        in_features=in_features,
        wiring=kncp.wirings.NCP(
            inter_neurons=16,
            command_neurons=8,
            motor_neurons=out_features,
            sensory_fanout=12,
            inter_fanout=4,
            recurrent_command_synapses=5,
            motor_fanin=8,
        ),
    ),
    CfC(
        in_features=in_features,
        hidden_size=32,
        out_features=out_features,
        use_mm_rnn=True,
    ),
    WiredCfC(
        in_features=in_features,
        wiring=kncp.wirings.FullyConnected(8, out_features),
        use_mm_rnn=True,
    ),
    WiredCfC(
        in_features=in_features,
        wiring=kncp.wirings.NCP(
            inter_neurons=16,
            command_neurons=8,
            motor_neurons=out_features,
            sensory_fanout=12,
            inter_fanout=4,
            recurrent_command_synapses=5,
            motor_fanin=8,
        ),
        use_mm_rnn=True,
    ),
]:
    dataloader = data.DataLoader(
        data.TensorDataset(data_x, data_y), batch_size=1, shuffle=True, num_workers=4
    )
    learn = SequenceLearner(model, lr=0.01)
    trainer = pl.Trainer(
        max_epochs=10,
        gradient_clip_val=1,  # Clip gradient to stabilize training
        gpus=1,
    )
    trainer.fit(learn, dataloader)