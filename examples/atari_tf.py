import tensorflow as tf
import gym
import ale_py
import numpy as np
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from ncps.tf import CfC
import numpy as np
from ncps.datasets.tf import AtariCloningDatasetTF


def run_closed_loop(model, env, num_episodes=None, rnn_to_reset=None):
    obs = env.reset()
    if rnn_to_reset is not None:
        rnn_to_reset.reset_states()
    returns = []
    total_reward = 0
    while True:
        # add batch and time dimension (with a single element in each)
        obs = np.expand_dims(np.expand_dims(obs, 0), 0)
        pred = model.predict(obs, verbose=0)
        action = pred[0, 0].argmax()  # remove time and batch dimension -> then argmax
        obs, r, done, _ = env.step(action)
        total_reward += r
        if done:
            returns.append(total_reward)
            total_reward = 0
            obs = env.reset()
            # Reset RNN hidden states when episode is over
            if rnn_to_reset is not None:
                rnn_to_reset.reset_states()
            if num_episodes is not None:
                # Count down the number of episodes
                num_episodes = num_episodes - 1
                if num_episodes == 0:
                    return returns


class ClosedLoopCallback(tf.keras.callbacks.Callback):
    def __init__(self, stateless_model, stateful_model, env, rnn_to_reset):
        self.stateless_model = stateless_model
        self.stateful_model = stateful_model
        self.env = env
        self.rnn_to_reset = rnn_to_reset

    def on_epoch_end(self, epoch, logs=None):
        # Copy weights from stateless model into stateful model
        self.stateful_model.set_weights(self.stateless_model.get_weights())
        r = run_closed_loop(
            self.stateful_model,
            self.env,
            num_episodes=10,
            rnn_to_reset=self.rnn_to_reset,
        )
        print(f"\nEpoch {epoch} return: {np.mean(r):0.2f} +- {np.std(r):0.2f}")


if __name__ == "__main__":

    env = gym.make("ALE/Breakout-v5")
    env = wrap_deepmind(env)

    data = AtariCloningDatasetTF("breakout")
    # batch size 32
    trainloader = data.get_dataset(32, split="train")
    valloader = data.get_dataset(32, split="val")

    conv_block = tf.keras.models.Sequential(
        [
            tf.keras.Input((84, 84, 4)),
            tf.keras.layers.Lambda(
                lambda x: tf.cast(x, tf.float32) / 255.0
            ),  # normalize input
            tf.keras.layers.Conv2D(64, 5, padding="same", activation="relu", strides=2),
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
    conv_block.build((None, 84, 84, 4))
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input((None, 84, 84, 4)),
            tf.keras.layers.TimeDistributed(conv_block),
            CfC(64, return_sequences=True, stateful=False),
            tf.keras.layers.Dense(env.action_space.n),
        ]
    )
    stateful_rnn = CfC(64, return_sequences=True, stateful=True)
    stateful_model = tf.keras.models.Sequential(
        [
            tf.keras.Input((None, 84, 84, 4), batch_size=1),
            tf.keras.layers.TimeDistributed(conv_block),
            stateful_rnn,
            tf.keras.layers.Dense(env.action_space.n),
        ]
    )

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.0001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.summary()
    model.fit(
        trainloader,
        epochs=50,
        validation_data=valloader,
        callbacks=[
            ClosedLoopCallback(model, stateful_model, env, rnn_to_reset=stateful_rnn)
        ],
    )

    # Visualize Atari game and play endlessly
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env = wrap_deepmind(env)
    run_closed_loop(model, env)