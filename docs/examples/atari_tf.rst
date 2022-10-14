Atari behavior cloning with TensorFlow
================================================

In this guide, we will train an NCP to play Atari.
Instead of learning a policy via reinforcement learning (which can be a bit complex), we will
make use of an pretrained expert policy that the NCP should copy using supervised learning (i.e., behavior cloning).

.. image:: ../img/breakout.webp
   :align: center


Setup and Requirements
-------------------------------------
Before we start, we need to install some packages

.. code-block:: bash

    pip3 install ncps tensorflow "ale-py==0.7.4" "ray[rllib]" "gym[atari,accept-rom-license]==0.23.1"

Defining the model
-------------------------------------
First, we will define the neural network model.
The model consists of a convolutional block, followed by a CfC recurrent neural network, and a final linear layer.

We first define a convolutional model that operates over just a batch of images, and then wrap it in a
``tf.keras.layers.TimeDistributed`` layer to apply the same convolutional block to a sequence of images.
When we apply the model in a closed-loop setting, we need some mechanisms to *remember* the hidden state, i.e., use the final hidden state of the previous data batch as the initial values of the hidden state for the current data batch.
This is implemented by implementing two different inference modes of the model:

#. A training mode, where we have a single input tensor (batch of sequences of images) and predicts a single output tensor.
#. A stateful mode, where the input and output are pairs, containing the initial state of the RNN and the final state of the RNN in the input and output as second element respectively.

.. code-block:: python

    import tensorflow as tf
    from ncps.tf import CfC
    import numpy as np

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

Dataloader
-------------------------------------
Next, we define the Atari environment and the dataset.
We have to wrap the environment with the helper functions proposed in `DeepMind's Atari Nature paper <https://www.nature.com/articles/nature14236>`_, which apply the following transformations:

* Downscales the Atari frames to 84-by-84 pixels
* Converts the frames to grayscale
* Stacks 4 consecutive frames into a single observation

The resulting observations are then 84-by-84 images with 4 channels.

For the behavior cloning dataset, we will use the ``AtariCloningDatasetTF`` dataset provided by the ``ncps`` package.

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



Running the model in a closed-loop
-------------------------------------
Next, we have to define the code for applying the model in a continuous control loop with the environment.
There are three subtleties we need to take care of:

#. Reset the RNN hidden states when a new episode starts in the Atari game
#. Reshape the input frames to have an extra batch and time dimension of size 1 as the network accepts only batches of sequences instead of single frames
#. Pack current hidden state together with the observation as input, and unpack the the prediction and next hidden state from the output

.. code-block:: python

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

Evaluating the closed-loop performance during training
----------------------------------------------------------
During the training, we measure only offline performance in the form of the training and validation accuracy.
However, we also want to check after every training epoch how the cloned network is performing when applied to the closed-loop environment.
To this end, we have to define a keras callback that is invoked after every training epoch and implements the closed-loop evaluation.


.. code-block:: python

    class ClosedLoopCallback(tf.keras.callbacks.Callback):
        def __init__(self, model, env):
            super().__init__()
            self.model = model
            self.env = env

        def on_epoch_end(self, epoch, logs=None):
            r = run_closed_loop(self.model, self.env, num_episodes=10)
            print(f"\nEpoch {epoch} return: {np.mean(r):0.2f} +- {np.std(r):0.2f}")



Training the model
-------------------------------------
Finally, we can instantiate the model and train it by using keras high-level ``model.fit`` functionality.

.. code-block:: python

    model = ConvCfC(env.action_space.n)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.0001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    # (batch, time, height, width, channels)
    model.build((None, None, 84, 84, 4))
    model.summary()

    model.fit(
        trainloader,
        epochs=50,
        validation_data=valloader,
        callbacks=[
            ClosedLoopCallback(model, env)
        ],
    )

After the training is completed we can display in a window how the model plays the game.

.. code-block:: python

    # Visualize Atari game and play endlessly
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env = wrap_deepmind(env)
    run_closed_loop(model, env)

The full source code can be downloaded `here <https://github.com/mlech26l/ncps/blob/master/examples/atari_tf.py>`_.

.. note::
    At a validation accuracy of about 92%, the behavior cloning data usually implies a decent closed-loop performance of the cloned agent.

The output of the full script is something like:

.. code-block:: text

    > Model: "sequential_1"
    > _________________________________________________________________
    >  Layer (type)                Output Shape              Param #
    > =================================================================
    >  time_distributed (TimeDistr  (None, None, 256)        1440576
    >  ibuted)
    >
    >  cf_c (CfC)                  (None, None, 64)          74112
    >
    >  dense (Dense)               (None, None, 4)           260
    >
    > =================================================================
    > Total params: 1,514,948
    > Trainable params: 1,514,948
    > Non-trainable params: 0
    > _________________________________________________________________
    > Epoch 1/50
    > 2022-10-11 15:45:55.524895: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8302
    > 2022-10-11 15:45:56.037075: I tensorflow/core/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
    > 938/938 [==============================] - ETA: 0s - loss: 0.4964 - sparse_categorical_accuracy: 0.8305
    > Epoch 0 return: 2.50 +- 1.91
    > 938/938 [==============================] - 413s 436ms/step - loss: 0.4964 - sparse_categorical_accuracy: 0.8305 - val_loss: 0.3931 - val_sparse_categorical_accuracy: 0.8633
    > Epoch 2/50
    > 938/938 [==============================] - ETA: 0s - loss: 0.3521 - sparse_categorical_accuracy: 0.8752
    > Epoch 1 return: 4.00 +- 3.58
    > 938/938 [==============================] - 450s 480ms/step - loss: 0.3521 - sparse_categorical_accuracy: 0.8752 - val_loss: 0.3168 - val_sparse_categorical_accuracy: 0.8884
    > Epoch 3/50
    > 938/938 [==============================] - ETA: 0s - loss: 0.3009 - sparse_categorical_accuracy: 0.8918
    > Epoch 2 return: 5.30 +- 3.32
    > 938/938 [==============================] - 469s 501ms/step - loss: 0.3009 - sparse_categorical_accuracy: 0.8918 - val_loss: 0.2732 - val_sparse_categorical_accuracy: 0.9020
    > Epoch 4/50
    > 938/938 [==============================] - ETA: 0s - loss: 0.2690 - sparse_categorical_accuracy: 0.9029
    > Epoch 3 return: 13.90 +- 9.54
    > 938/938 [==============================] - 514s 548ms/step - loss: 0.2690 - sparse_categorical_accuracy: 0.9029 - val_loss: 0.2485 - val_sparse_categorical_accuracy: 0.9103
    > Epoch 5/50
    > 938/938 [==============================] - ETA: 0s - loss: 0.2501 - sparse_categorical_accuracy: 0.9095
    > Epoch 4 return: 15.50 +- 14.33
    > 938/938 [==============================] - 516s 550ms/step - loss: 0.2501 - sparse_categorical_accuracy: 0.9095 - val_loss: 0.2475 - val_sparse_categorical_accuracy: 0.9107
    > Epoch 6/50
    > 938/938 [==============================] - ETA: 0s - loss: 0.2361 - sparse_categorical_accuracy: 0.9145
    > Epoch 5 return: 16.00 +- 12.49
    > 938/938 [==============================] - 514s 548ms/step - loss: 0.2361 - sparse_categorical_accuracy: 0.9145 - val_loss: 0.2363 - val_sparse_categorical_accuracy: 0.9150
    > Epoch 7/50
    > 938/938 [==============================] - ETA: 0s - loss: 0.2257 - sparse_categorical_accuracy: 0.9184
    > Epoch 6 return: 35.60 +- 30.20
    > 938/938 [==============================] - 508s 542ms/step - loss: 0.2257 - sparse_categorical_accuracy: 0.9184 - val_loss: 0.2256 - val_sparse_categorical_accuracy: 0.9183
    > Epoch 8/50
    > 938/938 [==============================] - ETA: 0s - loss: 0.2173 - sparse_categorical_accuracy: 0.9213
    > Epoch 7 return: 7.70 +- 5.59
    > 938/938 [==============================] - 501s 534ms/step - loss: 0.2173 - sparse_categorical_accuracy: 0.9213 - val_loss: 0.2179 - val_sparse_categorical_accuracy: 0.9207
    > Epoch 9/50
    > 938/938 [==============================] - ETA: 0s - loss: 0.2095 - sparse_categorical_accuracy: 0.9239
    > Epoch 8 return: 67.40 +- 81.60
    > 938/938 [==============================] - 555s 592ms/step - loss: 0.2095 - sparse_categorical_accuracy: 0.9239 - val_loss: 0.2045 - val_sparse_categorical_accuracy: 0.9265
    > Epoch 10/50
    > 938/938 [==============================] - ETA: 0s - loss: 0.2032 - sparse_categorical_accuracy: 0.9263
    > Epoch 9 return: 15.20 +- 12.16
    > 938/938 [==============================] - 523s 558ms/step - loss: 0.2032 - sparse_categorical_accuracy: 0.9263 - val_loss: 0.1962 - val_sparse_categorical_accuracy: 0.9290
    > Epoch 11/50
    > 938/938 [==============================] - ETA: 0s - loss: 0.1983 - sparse_categorical_accuracy: 0.9279
    > Epoch 10 return: 26.50 +- 27.98
    > 938/938 [==============================] - 512s 546ms/step - loss: 0.1983 - sparse_categorical_accuracy: 0.9279 - val_loss: 0.2180 - val_sparse_categorical_accuracy: 0.9210
    > Epoch 12/50
    > 938/938 [==============================] - ETA: 0s - loss: 0.1926 - sparse_categorical_accuracy: 0.9302
    > Epoch 11 return: 53.00 +- 79.22
    > 938/938 [==============================] - 1846s 2s/step - loss: 0.1926 - sparse_categorical_accuracy: 0.9302 - val_loss: 0.1924 - val_sparse_categorical_accuracy: 0.9311
    > ....