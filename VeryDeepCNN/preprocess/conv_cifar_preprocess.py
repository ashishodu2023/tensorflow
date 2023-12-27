from base.base_preprocess import BasePreProcess
import numpy as np
import tensorflow as tf


class ConvCifarPreProcess(BasePreProcess):
    def __init__(self, data, config):
        super(ConvCifarPreProcess, self).__init__(data, config)
        self.mean = np.mean(data[0], axis=(0, 1, 2, 3))
        self.std = np.std(data[0], axis=(0, 1, 2, 3))

    def get_normalized_train_data(self):
        self.X_train = (self.data[0] - self.mean) / (self.std + self.config.preprocess.epsilon)
        self.y_train = tf.keras.utils.to_categorical(self.data[1], 1000)
        return self.X_train, self.y_train

    def get_normalized_test_data(self):
        self.X_test = (self.X_test - self.mean) / (self.std + self.config.preprocess.epsilon)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, 1000)
        return self.X_train, self.y_train
