import argparse
import os

import gym
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPO
import time
import ale_py
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
import numpy as np

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override

import tensorflow as tf
from ncps.tf import CfC


class ConvCfCModel(RecurrentNetwork):
    """Example of using the Keras functional API to define a RNN model."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        cell_size=64,
    ):
        super(ConvCfCModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.cell_size = cell_size

        # Define input layers
        input_layer = tf.keras.layers.Input(
            shape=(None, obs_space.shape[0] * obs_space.shape[1] * obs_space.shape[2]),
            name="inputs",
        )
        state_in_h = tf.keras.layers.Input(shape=(cell_size,), name="h")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        # Preprocess observation with a hidden layer and send to CfC
        self.conv_block = tf.keras.models.Sequential(
            [
                tf.keras.Input(
                    (obs_space.shape[0] * obs_space.shape[1] * obs_space.shape[2])
                ),  # batch dimension is implicit
                tf.keras.layers.Lambda(
                    lambda x: tf.cast(x, tf.float32) / 255.0
                ),  # normalize input
                tf.keras.layers.Reshape(
                    (obs_space.shape[0], obs_space.shape[1], obs_space.shape[2])
                ),
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

        dense1 = self.td_conv(input_layer)
        cfc_out, state_h = CfC(
            cell_size, return_sequences=True, return_state=True, name="cfc"
        )(
            inputs=dense1,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h],
        )

        # Postprocess CfC output with another hidden layer and compute values
        logits = tf.keras.layers.Dense(
            self.num_outputs, activation=tf.keras.activations.linear, name="logits"
        )(cfc_out)
        values = tf.keras.layers.Dense(1, activation=None, name="values")(cfc_out)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h],
            outputs=[logits, values, state_h],
        )
        self.rnn_model.summary()

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h = self.rnn_model([inputs, seq_lens] + state)
        return model_out, [h]

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])


def run_closed_loop(algo, config):
    env = gym.make(args.env, render_mode="human")
    env = wrap_deepmind(env)
    rnn_cell_size = config["model"]["custom_model_config"]["cell_size"]
    obs = env.reset()
    state = init_state = [np.zeros(rnn_cell_size, np.float32)]
    while True:
        action, state, _ = algo.compute_single_action(
            obs, state=state, explore=False, policy_id="default_policy"
        )
        obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()
            state = init_state


ModelCatalog.register_custom_model("cfc", ConvCfCModel)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--cont", default="")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--hours", default=4, type=int)
    args = parser.parse_args()

    register_env("atari_env", lambda env_config: wrap_deepmind(gym.make(args.env)))
    config = {
        "env": "atari_env",
        "preprocessor_pref": None,
        "gamma": 0.99,
        "num_gpus": 1,
        "num_workers": 16,
        "num_envs_per_worker": 4,
        "create_env_on_driver": True,
        "lambda": 0.95,
        "kl_coeff": 0.5,
        "clip_rewards": True,
        "clip_param": 0.1,
        "vf_clip_param": 10.0,
        "entropy_coeff": 0.01,
        "rollout_fragment_length": 100,
        "sgd_minibatch_size": 500,
        "num_sgd_iter": 10,
        "batch_mode": "truncate_episodes",
        "observation_filter": "NoFilter",
        "model": {
            "vf_share_layers": True,
            "custom_model": "cfc",
            "max_seq_len": 20,
            "custom_model_config": {
                "cell_size": 64,
            },
        },
        "framework": "tf2",
    }

    algo = PPO(config=config)

    os.makedirs(f"rl_ckpt/{args.env}", exist_ok=True)
    if args.cont != "":
        algo.load_checkpoint(f"rl_ckpt/{args.env}/checkpoint-{args.cont}")

    if args.render:
        run_closed_loop(
            algo,
            config,
        )
    else:
        start_time = time.time()
        last_eval = 0
        while True:
            info = algo.train()
            if time.time() - last_eval > 60 * 5:  # every 5 minutes print some stats
                print(f"Ran {(time.time()-start_time)/60/60:0.1f} hours")
                print(
                    f"    sampled {info['info']['num_env_steps_sampled']/1000:0.0f}k steps"
                )
                print(f"    policy reward: {info['episode_reward_mean']:0.1f}")
                last_eval = time.time()
                ckpt = algo.save_checkpoint(f"rl_ckpt/{args.env}")
                print(f"    saved checkpoint '{ckpt}'")

            elapsed = (time.time() - start_time) / 60  # in minutes
            if elapsed > args.hours * 60:
                break