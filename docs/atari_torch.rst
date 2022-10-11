Playing Atari with an NCP and behavior cloning (PyTorch)
=========================

In this guide, we will train a NCP to play Atari.
Instead of learning a policy via reinforcement learning (which can be a bit complex), we will
make use of an pretrained expert policy that the NCP should copy using supervised learning (i.e., behavior cloning).

.. image:: ./img/breakout.webp
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
We have to wrap the environment with the Deepmind helper functions, which apply the following transformations:

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


The full source code can be downloaded `here <https://github.com/mlech26l/ncps/blob/master/examples/atari_torch.py>`_
When running the code we get

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
    > loss=0.1223: 100%|██████████| 938/938 [01:33<00:00, 10.03it/s]
    > Epoch 13, val_loss=0.5071, val_acc=84.78%
    > Mean return 0.5 (n=10)
    > loss=0.1176: 100%|██████████| 938/938 [01:33<00:00,  9.99it/s]
    > Epoch 14, val_loss=3.991, val_acc=23.81%
    > Mean return 0.7 (n=10)
    > loss=0.1134: 100%|██████████| 938/938 [01:33<00:00, 10.05it/s]
    > Epoch 15, val_loss=0.1758, val_acc=93.63%
    > Mean return 17.7 (n=10)
    > loss=0.1092: 100%|██████████| 938/938 [01:33<00:00,  9.99it/s]
    > Epoch 16, val_loss=0.4535, val_acc=83.59%
    > Mean return 3.6 (n=10)
    > loss=0.1058: 100%|██████████| 938/938 [01:33<00:00,  9.99it/s]
    > Epoch 17, val_loss=2.712, val_acc=37.89%
    > Mean return 0.5 (n=10)
    > loss=0.1018: 100%|██████████| 938/938 [01:32<00:00, 10.09it/s]
    > Epoch 18, val_loss=0.5907, val_acc=82.35%
    > Mean return 0.5 (n=10)
    > loss=0.0977: 100%|██████████| 938/938 [01:33<00:00,  9.98it/s]
    > Epoch 19, val_loss=0.5761, val_acc=80.92%
    > Mean return 3.0 (n=10)
    > loss=0.09423: 100%|██████████| 938/938 [01:33<00:00, 10.08it/s]
    > Epoch 20, val_loss=0.7243, val_acc=86.09%
    > Mean return 1.4 (n=10)
    > loss=0.09046: 100%|██████████| 938/938 [01:32<00:00, 10.14it/s]
    > Epoch 21, val_loss=0.2192, val_acc=92.44%
    > Mean return 27.2 (n=10)
    > loss=0.08728: 100%|██████████| 938/938 [01:32<00:00, 10.11it/s]
    > Epoch 22, val_loss=1.086, val_acc=66.81%
    > Mean return 4.0 (n=10)
    > loss=0.08372: 100%|██████████| 938/938 [01:33<00:00, 10.07it/s]
    > Epoch 23, val_loss=0.4594, val_acc=87.06%
    > Mean return 1.3 (n=10)
    > loss=0.08035: 100%|██████████| 938/938 [01:32<00:00, 10.14it/s]
    > Epoch 24, val_loss=0.9743, val_acc=67.74%
    > Mean return 2.7 (n=10)
    > loss=0.07681: 100%|██████████| 938/938 [01:32<00:00, 10.18it/s]
    > Epoch 25, val_loss=1.217, val_acc=63.96%
    > Mean return 3.4 (n=10)
    > loss=0.07353: 100%|██████████| 938/938 [01:32<00:00, 10.16it/s]
    > Epoch 26, val_loss=0.2653, val_acc=90.93%
    > Mean return 12.0 (n=10)
    > loss=0.07017: 100%|██████████| 938/938 [01:31<00:00, 10.23it/s]
    > Epoch 27, val_loss=0.3183, val_acc=89.67%
    > Mean return 10.8 (n=10)
    > loss=0.06709: 100%|██████████| 938/938 [01:32<00:00, 10.14it/s]
    > Epoch 28, val_loss=0.2179, val_acc=93.12%
    > Mean return 25.9 (n=10)
    > loss=0.06412: 100%|██████████| 938/938 [01:32<00:00, 10.19it/s]
    > Epoch 29, val_loss=0.5337, val_acc=87.40%
    > Mean return 1.3 (n=10)
    > loss=0.06137: 100%|██████████| 938/938 [01:32<00:00, 10.18it/s]
    > Epoch 30, val_loss=0.3089, val_acc=90.87%
    > Mean return 7.8 (n=10)
    > loss=0.05832: 100%|██████████| 938/938 [01:32<00:00, 10.15it/s]
    > Epoch 31, val_loss=0.246, val_acc=93.10%
    > Mean return 23.7 (n=10)
    > loss=0.05504: 100%|██████████| 938/938 [01:32<00:00, 10.15it/s]
    > Epoch 32, val_loss=0.2546, val_acc=92.98%
    > Mean return 62.1 (n=10)
    > loss=0.05302: 100%|██████████| 938/938 [01:32<00:00, 10.14it/s]
    > Epoch 33, val_loss=0.265, val_acc=92.27%
    > Mean return 13.3 (n=10)
    > loss=0.04998: 100%|██████████| 938/938 [01:32<00:00, 10.12it/s]
    > Epoch 34, val_loss=0.4808, val_acc=86.64%
    > Mean return 5.7 (n=10)
    > loss=0.04753: 100%|██████████| 938/938 [01:32<00:00, 10.15it/s]
    > Epoch 35, val_loss=2.868, val_acc=51.34%
    > Mean return 1.7 (n=10)
    > loss=0.0448: 100%|██████████| 938/938 [01:32<00:00, 10.19it/s]
    > Epoch 36, val_loss=2.086, val_acc=54.55%
    > Mean return 0.5 (n=10)
    > loss=0.04273: 100%|██████████| 938/938 [01:32<00:00, 10.15it/s]
    > Epoch 37, val_loss=0.4147, val_acc=89.81%
    > Mean return 5.2 (n=10)
    > loss=0.0408: 100%|██████████| 938/938 [01:32<00:00, 10.15it/s]
    > Epoch 38, val_loss=0.9393, val_acc=76.66%
    > Mean return 5.7 (n=10)
    > loss=0.03864: 100%|██████████| 938/938 [01:32<00:00, 10.15it/s]
    > Epoch 39, val_loss=0.2581, val_acc=92.52%
    > Mean return 69.7 (n=10)
    > loss=0.03636: 100%|██████████| 938/938 [01:32<00:00, 10.12it/s]
    > Epoch 40, val_loss=0.3293, val_acc=91.02%
    > Mean return 9.0 (n=10)
    > loss=0.03468: 100%|██████████| 938/938 [01:31<00:00, 10.20it/s]
    > Epoch 41, val_loss=0.2953, val_acc=91.73%
    > Mean return 19.5 (n=10)
    > loss=0.03316: 100%|██████████| 938/938 [01:32<00:00, 10.15it/s]
    > Epoch 42, val_loss=0.2843, val_acc=92.80%
    > Mean return 65.8 (n=10)
    > loss=0.03135: 100%|██████████| 938/938 [01:32<00:00, 10.12it/s]
    > Epoch 43, val_loss=0.2802, val_acc=92.41%
    > Mean return 7.5 (n=10)
    > loss=0.03014: 100%|██████████| 938/938 [01:32<00:00, 10.18it/s]
    > Epoch 44, val_loss=0.4413, val_acc=91.16%
    > Mean return 10.8 (n=10)
    > loss=0.02853: 100%|██████████| 938/938 [01:32<00:00, 10.17it/s]
    > Epoch 45, val_loss=0.2793, val_acc=92.84%
    > Mean return 60.8 (n=10)
    > loss=0.02736: 100%|██████████| 938/938 [01:32<00:00, 10.15it/s]
    > Epoch 46, val_loss=0.301, val_acc=92.97%
    > Mean return 17.3 (n=10)
    > loss=0.0259: 100%|██████████| 938/938 [01:32<00:00, 10.15it/s]
    > Epoch 47, val_loss=0.4634, val_acc=89.30%
    > Mean return 18.0 (n=10)
    > loss=0.02465: 100%|██████████| 938/938 [01:32<00:00, 10.15it/s]
    > Epoch 48, val_loss=0.4939, val_acc=89.54%
    > Mean return 2.3 (n=10)
    > loss=0.02412: 100%|██████████| 938/938 [01:32<00:00, 10.10it/s]
    > Epoch 49, val_loss=0.3671, val_acc=91.84%
    > Mean return 22.4 (n=10)
    > loss=0.02288: 100%|██████████| 938/938 [01:32<00:00, 10.17it/s]
    > Epoch 50, val_loss=0.2625, val_acc=93.67%
    > Mean return 43.3 (n=10)