:hide-toc:

===================================================
Welcome to Neural Circuit Policies's documentation!
===================================================

`Neural Circuit Policies (NCPs) <https://publik.tuwien.ac.at/files/publik_292280.pdf>`_ are machine learning models inspired by the nervous system of the nematode *C. elegans*.
This package provides easy-to-use implementations of NCPs for PyTorch and Tensorflow.

.. code-block:: bash

    pip3 install -U ncps

Example Pytorch example:

.. code-block:: python

    from ncps.torch import CfC

    # a fully connected CfC network
    rnn = CfC(input_size=20, units=50)
    x = torch.randn(2, 3, 20) # (batch, time, features)
    h0 = torch.zeros(2,50) # (batch, units)
    output, hn = rnn(x,h0)

A Tensorflow example

.. code-block:: python

    # Tensorflow example
    from ncps.tf import LTC
    from ncps.wirings import AutoNCP

    wiring = AutoNCP(28, 4) # 28 neurons, 4 outputs
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(None, 2)),
            # LTC model with NCP sparse wiring
            LTC(wiring, return_sequences=True),
        ]
    )



User’s Guide
--------------

.. toctree::
    :maxdepth: 2

    quickstart
    examples/index
    api/index