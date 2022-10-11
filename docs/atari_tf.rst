Playing Atari with an NCP and behavior cloning (TensorFlow)
=========================

In this guide, we will train a NCP to play Atari.
Instead of learning a policy via reinforcement learning (which can be a bit complex), we will
make use of an pretrained expert policy that the NCP should copy using supervised learning (i.e., behavior cloning).

.. image:: ../img/breakout.webp
   :align: center


Setup and Requirements
-------------------------------------
Before we start, we need to install some packages

.. code-block:: bash

    pip3 install ncps tensorflow ale-py==0.7.4 gym[atari,accept-rom-license]==0.23.1

Dataloader
-------------------------------------
First, we define the Atari environment and the dataset.
We have to wrap the environment with the Deepmind helper functions, which apply the following transformations:

* Downscales the Atari frames to 84-by-84 pixels
* Converts the frames to grayscale
* Stacks 4 consecutive frames into a single observation

The resulting observations are then 84-by-84 images with 4 channels.

For the behavior cloning dataset, we will use the `AtariCloningDatasetTF` dataset provided by the `ncps` package.

.. code-block:: python

    import gym
    import ale_py
    from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
    from ncps.datasets.tf import AtariCloningDatasetTF

    env = gym.make("ALE/Breakout-v5")
    # We need to wrap the environment with the Deepmind helper functions
    env = wrap_deepmind(env)

    data = AtariCloningDatasetTF("breakout")
    # batch size 32
    trainloader = data.get_dataset(32, split="train")
    valloader = data.get_dataset(32, split="val")


Defining the model
-------------------------------------
Next, we will define the neural network model.
The model consists of a convolutional block, followed by a CfC recurrent neural network.

We first define a convolutional model that operates over just a batch of images, and then wrap it in a
`tf.keras.layers.TimeDistributed` layer to apply the same convolutional block to a sequence of images.
The final model operates not on a batch of images but a batch of sequences of images, thus having an extra dimension.

.. code-block:: python

    conv_block = tf.keras.models.Sequential(
        [
            tf.keras.Input((84, 84, 4)),
            tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0), # normalize input
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
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.0001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.summary()


Defining a stateful model
-------------------------------------
The model we defined above operates on sequences.

.. code-block:: python

    stateful_rnn = CfC(64, return_sequences=True, stateful=True)
    stateful_model = tf.keras.models.Sequential(
        [
            tf.keras.Input((None, 84, 84, 4), batch_size=1),
            tf.keras.layers.TimeDistributed(conv_block),
            stateful_rnn,
            tf.keras.layers.Dense(env.action_space.n),
        ]
    )


Defining the model
-------------------------------------

.. code-block:: python

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


Defining the model
-------------------------------------

.. code-block:: python

    model.fit(
        trainloader,
        epochs=50,
        validation_data=valloader,
        callbacks=[
            ClosedLoopCallback(model, stateful_model, env, rnn_to_reset=stateful_rnn)
        ],
    )

trains an MLP with one hidden layer to a maximum reward of 500 in less than a second


The full source code can be downloaded `here <todo>`_