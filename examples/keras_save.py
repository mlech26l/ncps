import numpy as np
import tensorflow as tf

data_x = np.random.default_rng().normal(size=(100, 16, 10))
data_y = np.random.default_rng().normal(size=(100, 16, 1))


@tf.keras.utils.register_keras_serializable(package="Custom")
class CustomLayer(tf.keras.layers.AbstractRNNCell):
    def __init__(self, units, outputs, **kwargs):
        self.units = units
        self.outputs = outputs
        super(CustomLayer, self).__init__(**kwargs)

    @property
    def state_size(self):
        return self.units

    def build(self, input_shape):
        self.w = self.add_weight("w", shape=(input_shape[-1], self.units))
        self.o = self.add_weight("o", shape=(self.units, self.outputs))
        self.r = self.add_weight("r", shape=(self.units, self.units))
        self.built = True

    def call(self, inputs, states):
        next_hidden = tf.nn.tanh(
            tf.matmul(inputs, self.w) + tf.matmul(states[0], self.r)
        )
        output = tf.matmul(next_hidden, self.o)
        return output, [next_hidden]

    def get_config(self):
        return {"units": self.units, "outputs": self.outputs}


model = tf.keras.models.Sequential(
    [
        tf.keras.Input((16, 10,)),
        tf.keras.layers.RNN(CustomLayer(10, 1), return_sequences=True),
    ]
)
model.compile(optimizer="adam", loss="mean_squared_error")

model.summary()

model.fit(x=data_x, y=data_y, batch_size=25, epochs=1)
model.evaluate(x=data_x, y=data_y)

model.save("test_2.h5")

model = tf.keras.models.load_model("test_2.h5")  ## Crashes here
model.evaluate(x=data_x, y=data_y)
