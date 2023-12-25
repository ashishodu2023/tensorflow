import tensorflow as tf
from CNN.LeNet.BaseModel import BaseModel
import matplotlib.pyplot as plt
import numpy as np

class Configuration:
    IMG_CHANNEL = 3
    IMG_ROW, IMG_COL = 32, 32  # input dimensions
    EPOCHS = 20
    BATCH_SIZE = 128
    VERBOSE = 1
    OPTIMIZER = tf.keras.optimizers.RMSprop()
    VALIDATION_SPLIT = 0.9
    INPUT_SHAPE = (IMG_ROW, IMG_COL, IMG_CHANNEL)
    NB_CLASSES = 10  # number of outputs
    TRAIN_SET = 50000
    TEST_SET = 10000
    DROPOUT = 0.5
    RELU = 'relu'
    SOFTMAX = 'softmax'
    LOSS = 'categorical_crossentropy'
    METRICS = 'accuracy'
    FLOAT = 'float32'


def plot_loss(history):
    plt.plot(history.history['loss'],label='loss')
    plt.plot(history.history['val_loss'],label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.show()


class Cifar10BaseCNN(BaseModel):
    def __init__(self):
        self.score = None
        self.callbacks = None
        self.history = None
        self.model = None

    def load_data(self):
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        return (x_train, y_train), (x_test, y_test)

    def preprocess(self, x_train, y_train, x_test, y_test):
        # Normalize the data
        x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], Configuration.IMG_CHANNEL)
        x_train=x_train / 255.0
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], Configuration.IMG_CHANNEL)
        x_test=x_test / 255.0
        x_train = x_train.astype(Configuration.FLOAT)
        x_test = x_test.astype(Configuration.FLOAT)

        y_train = tf.keras.utils.to_categorical(y_train, Configuration.NB_CLASSES)
        y_test = tf.keras.utils.to_categorical(y_test, Configuration.NB_CLASSES)

        return x_train, y_train, x_test, y_test

    def build_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Conv2D(
            32, (3, 3), activation=Configuration.RELU, input_shape=Configuration.INPUT_SHAPE
        ))
        self.model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(512, activation=Configuration.RELU))
        self.model.add(tf.keras.layers.Dropout(Configuration.DROPOUT))
        self.model.add(tf.keras.layers.Dense(Configuration.NB_CLASSES, activation=Configuration.SOFTMAX))
        self.model.build(input_shape=Configuration.INPUT_SHAPE)
        self.model.compile(loss=Configuration.LOSS, optimizer=Configuration.OPTIMIZER, metrics=[Configuration.METRICS])
        return self.model

    def train_model(self, train_features, train_label):
        self.callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir='logs')
        ]
        self.history = self.model.fit(train_features, train_label, batch_size=Configuration.BATCH_SIZE,
                                      epochs=Configuration.EPOCHS,
                                      verbose=Configuration.VERBOSE, validation_split=Configuration.VALIDATION_SPLIT,
                                      callbacks=self.callbacks)
        return self.history

    def predictions(self,model,x_test,y_test):
        self.score = model.evaluate(x_test,y_test,verbose=Configuration.VERBOSE)
        print("\n Test Score:",self.score[0])
        print("\n Test Accuracy:",self.score[1])

