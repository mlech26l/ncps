<div align="center"><img src="https://raw.githubusercontent.com/mlech26l/ncps/master/docs/img/banner.png" width="800"/></div>

# Neural Circuit Policies (for PyTorch and TensorFlow)

[![DOI](https://zenodo.org/badge/290199641.svg)](https://zenodo.org/badge/latestdoi/290199641)
![ci_badge](https://github.com/mlech26l/ncps/actions/workflows/python-test.yml/badge.svg) 
![pyversion](misc/pybadge.svg)
![PyPI version](https://img.shields.io/pypi/v/ncps)
![Documentation Status](https://readthedocs.org/projects/ncps/badge/?version=latest)
![downloads](https://img.shields.io/pypi/dm/ncps)

## 📜 Papers

[Neural Circuit Policies Enabling Auditable Autonomy (Open Access)](https://publik.tuwien.ac.at/files/publik_292280.pdf).  
[Closed-form continuous-time neural networks (Open Access)](https://www.nature.com/articles/s42256-022-00556-7)

Neural Circuit Policies (NCPs) are designed sparse recurrent neural networks loosely inspired by the nervous system of the organism [C. elegans](http://www.wormbook.org/chapters/www_celegansintro/celegansintro.html). 
The goal of this package is to making working with NCPs in PyTorch and keras as easy as possible.

[📖 Docs](https://ncps.readthedocs.io/en/latest/index.html)

```python
import torch
from ncps.torch import CfC

rnn = CfC(20,50) # (input, hidden units)
x = torch.randn(2, 3, 20) # (batch, time, features)
h0 = torch.zeros(2,50) # (batch, units)
output, hn = rnn(x,h0)
```


## Installation

```bash
pip install ncps
```

## 🔖 Colab Notebooks

We have created a few Google Colab notebooks for an interactive introduction to the package

- [Google Colab (Pytorch) Basic usage](https://colab.research.google.com/drive/1VWoGcpyqGvrUOUzH7ccppE__m-n1cAiI?usp=sharing)
- [Google Colab (Tensorflow): Basic usage](https://colab.research.google.com/drive/1IvVXVSC7zZPo5w-PfL3mk1MC3PIPw7Vs?usp=sharing)
- [Google Colab (Tensorflow): Processing irregularly sampled time-series](https://colab.research.google.com/drive/1wBojTMMMVWl2WbF6hASbST1-XhK_xs5u?usp=sharing)
- [Google Colab (Tensorflow) Stacking NCPs with other layers](https://colab.research.google.com/drive/1-mZunxqVkfZVBXNPG0kTSKUNQUSdZiBI?usp=sharing)

## End-to-end Examples

- [Quickstart (torch and tf)](https://ncps.readthedocs.io/en/latest/quickstart.html)
- [Atari Behavior Cloning (torch and tf)](https://ncps.readthedocs.io/en/latest/examples/atari_bc.html)
- [Atari Reinforcement Learning (tf)](https://ncps.readthedocs.io/en/latest/examples/atari_ppo.html)

## Usage: Models and Wirings

The package provides two models, the liquid time-constant (LTC) and the closed-form continuous-time (CfC) models.
Both models are available as ```tf.keras.layers.Layer``` or ```torch.nn.Module``` RNN layers.

```python
from ncps.torch import CfC, LTC

input_size = 20
units = 28 # 28 neurons
rnn = CfC(input_size, units)
rnn = LTC(input_size, units)
```

The RNNs defined above consider fully-connected layers, i.e., as in LSTM, GRUs, and other RNNs.
The distinctiveness of NCPs is their structured wiring diagram. 
To combine the LTC or CfC model with a 

```python
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP

wiring = AutoNCP(28, 4) # 28 neurons, 4 outputs
input_size = 20
rnn = CfC(input_size, wiring)
rnn = LTC(input_size, wiring)
```

![alt](https://github.com/mlech26l/ncps/raw/master/docs/img/things.png)

## Tensorflow

The Tensorflow bindings are available via the ```ncps.tf``` module.

```python
from ncps.tf import CfC, LTC
from ncps.wirings import AutoNCP

units = 28
wiring = AutoNCP(28, 4) # 28 neurons, 4 outputs
input_size = 20
rnn1 = LTC(units) # fully-connected LTC
rnn2 = CfC(units) # fully-connected CfC
rnn3 = LTC(wiring) # NCP wired LTC
rnn4 = CfC(wiring) # NCP wired CfC
```

We can then combine the NCP cell with arbitrary ```tf.keras.layers```, for instance to build a powerful image sequence classifier:

```python
from ncps.wirings import AutoNCP
from ncps.tf import LTC
import tensorflow as tf
height, width, channels = (78, 200, 3)

ncp = LTC(AutoNCP(32, output_size=8), return_sequences=True)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(None, height, width, channels)),
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(32, (5, 5), activation="relu")
        ),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D()),
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(64, (5, 5), activation="relu")
        ),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D()),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation="relu")),
        ncp,
        tf.keras.layers.TimeDistributed(tf.keras.layers.Activation("softmax")),
    ]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss='sparse_categorical_crossentropy',
)
```

```bib
@article{lechner2020neural,
  title={Neural circuit policies enabling auditable autonomy},
  author={Lechner, Mathias and Hasani, Ramin and Amini, Alexander and Henzinger, Thomas A and Rus, Daniela and Grosu, Radu},
  journal={Nature Machine Intelligence},
  volume={2},
  number={10},
  pages={642--652},
  year={2020},
  publisher={Nature Publishing Group}
}
```
