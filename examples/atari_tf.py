# Copyright 2022 Mathias Lechner
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

import tensorflow as tf
import gym
import ale_py
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from ncps.tf import CfC
import numpy as np
from ncps.datasets.tf import AtariCloningDatasetTF


class ConvCfC(tf.keras.Model):
    def __init__(self, n_actions):
        super().__init__()
        self.conv_block = tf.keras.models.Sequential(
            [
                tf.keras.Input((84, 84, 4)),
                tf.keras.layers.Lambda(
                    lambda x: tf.cast(x, tf.float32) / 255.0
                ),  # normalize input
                tf.keras.layers.Conv2D(
                    64, 5, padding="same", activation="relu", strides=2
                ),
                tf.keras.layers.Conv2D(
                    128, 5, padding="same", activation="relu", strides=2
                ),
                tf.keras.layers.Conv2D(
                    128, 5, padding="same", activation="relu", strides=2
                ),
                tf.keras.layers.Conv2D(
                    256, 5, padding="same", activation="relu", strides=2
                ),
                tf.keras.layers.GlobalAveragePooling2D(),
            ]
        )
        self.td_conv = tf.keras.layers.TimeDistributed(self.conv_block)
        self.rnn = CfC(64, return_sequences=True, return_state=True)
        self.linear = tf.keras.layers.Dense(n_actions)

    def get_initial_states(self, batch_size=1):
        return self.rnn.cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)

    def call(self, x, training=None, **kwargs):
        has_hx = isinstance(x, list) or isinstance(x, tuple)
        initial_state = None
        if has_hx:
            # additional inputs are passed as a tuple
            x, initial_state = x

        x = self.td_conv(x, training=training)
        x, next_state = self.rnn(x, initial_state=initial_state)
        x = self.linear(x)
        if has_hx:
            return (x, next_state)
        return x


def run_closed_loop(model, env, num_episodes=None):
    obs = env.reset()
    hx = model.get_initial_states()
    returns = []
    total_reward = 0
    while True:
        # add batch and time dimension (with a single element in each)
        obs = np.expand_dims(np.expand_dims(obs, 0), 0)
        pred, hx = model.predict((obs, hx), verbose=0)
        action = pred[0, 0].argmax()
        # remove time and batch dimension -> then argmax
        obs, r, done, _ = env.step(action)
        total_reward += r
        if done:
            returns.append(total_reward)
            total_reward = 0
            obs = env.reset()
            hx = model.get_initial_states()
            # Reset RNN hidden states when episode is over
            if num_episodes is not None:
                # Count down the number of episodes
                num_episodes = num_episodes - 1
                if num_episodes == 0:
                    return returns


class ClosedLoopCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, env):
        super().__init__()
        self.model = model
        self.env = env

    def on_epoch_end(self, epoch, logs=None):
        r = run_closed_loop(self.model, self.env, num_episodes=10)
        print(f"\nEpoch {epoch} return: {np.mean(r):0.2f} +- {np.std(r):0.2f}")


if __name__ == "__main__":

    env = gym.make("ALE/Breakout-v5")
    env = wrap_deepmind(env)

    data = AtariCloningDatasetTF("breakout")
    # batch size 32
    trainloader = data.get_dataset(32, split="train")
    valloader = data.get_dataset(32, split="val")

    model = ConvCfC(env.action_space.n)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.0001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.build((None, None, 84, 84, 4))
    model.summary()
    model.fit(
        trainloader,
        epochs=50,
        validation_data=valloader,
        callbacks=[ClosedLoopCallback(model, env)],
    )

    # Visualize Atari game and play endlessly
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env = wrap_deepmind(env)
    run_closed_loop(model, env)