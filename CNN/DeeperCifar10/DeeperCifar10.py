import tensorflow as tf
from BaseModel import BaseModel
import matplotlib.pyplot as plt
import numpy as np


class Configuration:
    IMG_CHANNEL = 3
    IMG_ROW, IMG_COL = 32, 32  # input dimensions
    EPOCHS = 50
    BATCH_SIZE = 128
    VERBOSE = 1
    OPTIMIZER = tf.keras.optimizers.RMSprop()
    VALIDATION_SPLIT = 0.9
    INPUT_SHAPE = (IMG_ROW, IMG_COL, IMG_CHANNEL)
    NB_CLASSES = 10  # number of outputs
    TRAIN_SET = 50000
    TEST_SET = 10000
    DROPOUT_1 = 0.2
    DROPOUT_2 = 0.3
    DROPOUT_3 = 0.4
    EPSILON = 1e-7
    RELU = 'relu'
    SOFTMAX = 'softmax'
    LOSS = 'categorical_crossentropy'
    METRICS = 'accuracy'
    FLOAT = 'float32'


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.show()


class DeeperCifar10CNN(BaseModel):
    def __init__(self):
        self.mean = 0.0
        self.std = 0.0
        self.score = None
        self.callbacks = None
        self.history = None
        self.model = None

    def load_data(self):
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        return (x_train, y_train), (x_test, y_test)

    def preprocess(self, x_train, y_train, x_test, y_test):

        x_train = x_train.astype(Configuration.FLOAT)
        x_test = x_test.astype(Configuration.FLOAT)

        #Normalize
        self.mean=np.mean(x_train,axis=(0,1,2,3))
        self.std=np.std(x_train,axis=(0,1,2,3))
        x_train = (x_train-self.mean)/(self.std+Configuration.EPSILON)
        x_test = (x_test-self.mean)/(self.std+Configuration.EPSILON)
        y_train = tf.keras.utils.to_categorical(y_train, Configuration.NB_CLASSES)
        y_test = tf.keras.utils.to_categorical(y_test, Configuration.NB_CLASSES)
        return x_train, y_train, x_test, y_test

    def build_model(self,input_shape):
        self.model = tf.keras.models.Sequential()

        # 1st Block
        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',input_shape=input_shape,
                                              activation=Configuration.RELU))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                                              activation=Configuration.RELU))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Dropout(Configuration.DROPOUT_1))

        # 2nd Block
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same',input_shape=input_shape,
                                              activation=Configuration.RELU))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                                              activation=Configuration.RELU))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Dropout(Configuration.DROPOUT_2))

        # 3rd Block
        self.model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same',input_shape=input_shape,
                                              activation=Configuration.RELU))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same',
                                              activation=Configuration.RELU))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Dropout(Configuration.DROPOUT_3))

        # Dense
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(Configuration.NB_CLASSES, activation=Configuration.SOFTMAX))
        #self.model.build(input_shape=input_shape)
        self.model.compile(loss=Configuration.LOSS, optimizer=Configuration.OPTIMIZER, metrics=[Configuration.METRICS])
        return self.model

    def train_model(self, train_features, train_label,test_features,test_label):
        self.history = self.model.fit(train_features, train_label, batch_size=Configuration.BATCH_SIZE,
                                      epochs=Configuration.EPOCHS,
                                      verbose=Configuration.VERBOSE, validation_data=(test_features,test_label)
                                      )
        return self.history

    def predictions(self, model, x_test, y_test):
        self.score = model.evaluate(x_test, y_test, verbose=Configuration.VERBOSE)
        print("\n Test Score:", self.score[0])
        print("\n Test Accuracy:", self.score[1])
