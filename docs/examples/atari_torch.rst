Atari behavior cloning with PyTorch
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

    pip3 install ncps torch ale-py==0.7.4 gym[atari,accept-rom-license]==0.23

Defining the model
-------------------------------------
The model consists of a convolutional block, followed by a CfC recurrent neural network.
We first define a convolutional model that operates over just a batch of images and outputs a feature vector.

.. code-block:: python

    import torch.nn as nn
    import torch.nn.functional as F

    class ConvBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(4, 64, 5, padding=2, stride=2)
            self.conv2 = nn.Conv2d(64, 128, 5, padding=2, stride=2)
            self.conv3 = nn.Conv2d(128, 128, 5, padding=2, stride=2)
            self.conv4 = nn.Conv2d(128, 256, 5, padding=2, stride=2)
            self.norm = nn.BatchNorm1d(256)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = x.mean((-1, -2))  # Global average pooling
            x = self.norm(x)
            return x

Next, we define a sequence model that first applies the same convolutional block to a sequence of images, followed by a CfC recurrent neural network.

.. code-block:: python

    from ncps.torch import CfC
    class ConvCfC(nn.Module):
        def __init__(self, n_actions):
            super().__init__()
            self.conv_block = ConvBlock()
            self.rnn = CfC(256, 64, batch_first=True, proj_size=n_actions)

        def forward(self, x, hx=None):
            batch_size = x.size(0)
            seq_len = x.size(1)
            # Merge time and batch dimension into a single one (because the Conv layers require this)
            x = x.view(batch_size * seq_len, *x.shape[2:])
            x = self.conv_block(x)  # apply conv block to merged data
            # Separate time and batch dimension again
            x = x.view(batch_size, seq_len, *x.shape[1:])
            x, hx = self.rnn(x, hx)  # hx is the hidden state of the RNN
            return x, hx

Dataloader
-------------------------------------
Here, we define the Atari environment and the dataset.
We have to wrap the environment with the helper functions proposed in `DeepMind's Atari Nature paper <https://www.nature.com/articles/nature14236>`_, which apply the following transformations:

* Downscales the Atari frames to 84-by-84 pixels
* Converts the frames to grayscale
* Stacks 4 consecutive frames into a single observation

The resulting observations are then 84-by-84 images with 4 channels.

For the behavior cloning dataset, we will use the `AtariCloningDataset` PyTorch dataset provided by the `ncps` package.

.. code-block:: python

    import gym
    import ale_py
    import torch
    from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
    from torch.utils.data import Dataset
    import torch.optim as optim

    from ncps.datasets.torch import AtariCloningDataset

    env = gym.make("ALE/Breakout-v5")
    # We need to wrap the environment with the Deepmind helper functions
    env = wrap_deepmind(env)

    train_ds = AtariCloningDataset("breakout", split="train")
    val_ds = AtariCloningDataset("breakout", split="val")
    trainloader = torch.utils.data.DataLoader(
        train_ds, batch_size=32, num_workers=4, shuffle=True
    )
    valloader = torch.utils.data.DataLoader(val_ds, batch_size=32, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvCfC(n_actions=env.action_space.n).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

Training loop
-------------------------------------
For the training, we define a function that train the model by making one pass over the dataset.

.. code-block:: python

    def train_one_epoch(model, criterion, optimizer, trainloader):
        running_loss = 0.0
        pbar = tqdm(total=len(trainloader))
        model.train()
        device = next(model.parameters()).device  # get device the model is located on
        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)  # move data to same device as the model
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs, hx = model(inputs)
            labels = labels.view(-1, *labels.shape[2:])  # flatten
            outputs = outputs.reshape(-1, *outputs.shape[2:])  # flatten
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            pbar.set_description(f"loss={running_loss / (i + 1):0.4g}")
            pbar.update(1)
        pbar.close()

We also want to track the offline performance (= accuracy) of the model on the validation set.
To this end, we define another function that iterates over a dataset and measures the accuracy.

.. code-block:: python

    def eval(model, valloader):
        losses, accs = [], []
        model.eval()
        device = next(model.parameters()).device  # get device the model is located on
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs = inputs.to(device)  # move data to same device as the model
                labels = labels.to(device)

                outputs, _ = model(inputs)
                outputs = outputs.reshape(-1, *outputs.shape[2:])  # flatten
                labels = labels.view(-1, *labels.shape[2:])  # flatten
                loss = criterion(outputs, labels)
                acc = (outputs.argmax(-1) == labels).float().mean()
                losses.append(loss.item())
                accs.append(acc.item())
        return np.mean(losses), np.mean(accs)


Running the model in a closed-loop
-------------------------------------
Next, we have to define the code for applying the model in a continuous control loop with the environment.
There are two subtleties we need to take care of:

#. Reset the RNN hidden states when a new episode starts in the Atari game
#. Reshape the input frames to have an extra batch and time dimension of size 1 as the network accepts only batches of sequences instead of single frames

.. code-block:: python

    def run_closed_loop(model, env, num_episodes=None):
        obs = env.reset()
        device = next(model.parameters()).device
        hx = None  # Hidden state of the RNN
        returns = []
        total_reward = 0
        with torch.no_grad():
            while True:
                # PyTorch require channel first images -> transpose data
                obs = np.transpose(obs, [2, 0, 1]).astype(np.float32) / 255.0
                # add batch and time dimension (with a single element in each)
                obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(device)
                pred, hx = model(obs, hx)
                # remove time and batch dimension -> then argmax
                action = pred.squeeze(0).squeeze(0).argmax().item()
                obs, r, done, _ = env.step(action)
                total_reward += r
                if done:
                    obs = env.reset()
                    hx = None  # Reset hidden state of the RNN
                    returns.append(total_reward)
                    total_reward = 0
                    if num_episodes is not None:
                        # Count down the number of episodes
                        num_episodes = num_episodes - 1
                        if num_episodes == 0:
                            return returns


Training the model
-------------------------------------
With the functions and model defined above, we can how implement our training procedure very conveniently.

.. code-block:: python

    for epoch in range(50):  # loop over the dataset multiple times
        train_one_epoch(model, criterion, optimizer, trainloader)

        # Evaluate model on the validation set
        val_loss, val_acc = eval(model, valloader)
        print(f"Epoch {epoch+1}, val_loss={val_loss:0.4g}, val_acc={100*val_acc:0.2f}%")

        # Apply model in closed-loop environment
        returns = run_closed_loop(model, env, num_episodes=10)
        print(f"Mean return {np.mean(returns)} (n={len(returns)})")

After the training is completed we can display in a window how the model plays the game.

.. code-block:: python

    # Visualize Atari game and play endlessly
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env = wrap_deepmind(env)
    run_closed_loop(model, env)

The full source code can be downloaded `here <https://github.com/mlech26l/ncps/blob/master/examples/atari_torch.py>`_

.. note::
    At a validation accuracy of about 92% the behavior cloning data usually implies a decent closed-loop performance of the cloned agent

The output of the full script is something like:

.. code-block:: text

    > loss=0.4349: 100%|██████████| 938/938 [01:35<00:00,  9.83it/s]
    > Epoch 1, val_loss=1.67, val_acc=31.94%
    > Mean return 0.2 (n=10)
    > loss=0.2806: 100%|██████████| 938/938 [01:30<00:00, 10.33it/s]
    > Epoch 2, val_loss=0.43, val_acc=83.51%
    > Mean return 3.7 (n=10)
    > loss=0.223: 100%|██████████| 938/938 [01:31<00:00, 10.30it/s]
    > Epoch 3, val_loss=0.2349, val_acc=91.43%
    > Mean return 4.9 (n=10)
    > loss=0.1951: 100%|██████████| 938/938 [01:31<00:00, 10.26it/s]
    > Epoch 4, val_loss=2.824, val_acc=29.19%
    > Mean return 0.6 (n=10)
    > loss=0.1786: 100%|██████████| 938/938 [01:30<00:00, 10.33it/s]
    > Epoch 5, val_loss=0.3122, val_acc=89.03%
    > Mean return 4.0 (n=10)
    > loss=0.1669: 100%|██████████| 938/938 [01:31<00:00, 10.22it/s]
    > Epoch 6, val_loss=4.272, val_acc=22.84%
    > Mean return 0.5 (n=10)
    > loss=0.1575: 100%|██████████| 938/938 [01:32<00:00, 10.14it/s]
    > Epoch 7, val_loss=0.2788, val_acc=89.78%
    > Mean return 9.9 (n=10)
    > loss=0.15: 100%|██████████| 938/938 [01:33<00:00, 10.08it/s]
    > Epoch 8, val_loss=3.725, val_acc=25.07%
    > Mean return 0.6 (n=10)
    > loss=0.1429: 100%|██████████| 938/938 [01:31<00:00, 10.23it/s]
    > Epoch 9, val_loss=0.5851, val_acc=77.82%
    > Mean return 44.6 (n=10)
    > loss=0.1369: 100%|██████████| 938/938 [01:32<00:00, 10.12it/s]
    > Epoch 10, val_loss=0.7148, val_acc=71.74%
    > Mean return 3.4 (n=10)
    > loss=0.1316: 100%|██████████| 938/938 [01:32<00:00, 10.11it/s]
    > Epoch 11, val_loss=0.2138, val_acc=92.27%
    > Mean return 15.8 (n=10)
    > loss=0.1267: 100%|██████████| 938/938 [01:33<00:00, 10.02it/s]
    > Epoch 12, val_loss=0.2683, val_acc=90.54%
    > Mean return 14.3 (n=10)
    > ....