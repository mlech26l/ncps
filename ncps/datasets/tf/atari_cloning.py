from pathlib import Path
import tensorflow as tf
import os
import numpy as np
from ncps.datasets.utils import download_and_unzip


def py_load(x):
    x = x.numpy().decode("UTF-8")
    arr = np.load(x)
    x = arr["obs"]
    y = arr["actions"]
    return tf.convert_to_tensor(x), tf.convert_to_tensor(y)


def load_fn(x):
    x, y = tf.py_function(func=py_load, inp=[x], Tout=[tf.uint8, tf.int64])
    x = tf.ensure_shape(x, (32, 84, 84, 4))
    y = tf.ensure_shape(y, (32))
    return x, y


class AtariCloningDatasetTF:
    def __init__(self, env_name, root_dir="."):
        path = Path(root_dir) / "data_atari_seq" / env_name
        if not path.exists():
            print("Downloading data ... ", end="", flush=True)
            download_and_unzip(
                f"https://people.csail.mit.edu/mlechner/datasets/{env_name}.zip"
            )
            print("[done]")

        self.val_files = [str(s) for s in list(path.glob("val_*.npz"))]
        self.train_files = [str(s) for s in list(path.glob("train_*.npz"))]
        if len(self.val_files) == 0 or len(self.train_files) == 0:
            raise RuntimeError("Could not find data")

    def get_dataset(self, batch_size, split):
        if split == "train":
            dataset = tf.data.Dataset.from_tensor_slices(self.train_files)
            dataset = dataset.shuffle(30000)
        elif split == "val":
            dataset = tf.data.Dataset.from_tensor_slices(self.val_files)
        else:
            raise ValueError(
                f"Invalid split '{split}', must be either 'train' or 'val'"
            )
        dataset = dataset.map(load_fn)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(10)
        return dataset