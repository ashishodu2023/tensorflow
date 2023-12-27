from base.base_data_loader import BaseDataLoader
from keras.datasets import cifar10
import numpy as np
import tensorflow as tf


class ConvCifarDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(ConvCifarDataLoader, self).__init__(config)
        self.std = None
        self.mean = None
        (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data()
        # self.X_train = self.X_train.reshape((-1, 32, 32, 1))
        # self.X_test = self.X_test.reshape((-1, 32, 32, 1))

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_normalized_data(self):
        self.mean = np.mean(self.X_train, axis=(0, 1, 2, 3))
        self.std = np.std(self.X_train, axis=(0, 1, 2, 3))
        self.X_train = (self.X_train - self.mean) / (self.std + 1e-7)
        self.X_test = (self.X_test - self.mean) / (self.std + 1e-7)
        self.y_train = tf.keras.utils.to_categorical(self.y_train, 10)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, 10)
        return self.X_train, self.y_train
