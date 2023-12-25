import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten
from tensorflow import keras
from CNN.BaseModel import BaseModel
from utils import PlotDigits
import warnings
warnings.filterwarnings('ignore')

# Network and training parameters


class Configuration:
    EPOCHS = 50
    VERBOSE = 1
    OPTIMIZER = 'adam'
    METRICS = 'accuracy'
    LOSS = 'sparse_categorical_crossentropy'


class HandWritingClassification(BaseModel):

    def __init__(self):
        self.model = None

    def load_data(self):
        mnist = keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return (x_train, y_train), (x_test, y_test)

    def preprocess(self, x_train, y_train, x_test, y_test):
        # Normalize the data
        x_train = x_train/np.float32(255)
        y_train = y_train.astype(np.int32)
        x_test = x_test/np.float32(255)
        y_test = y_test.astype(np.int32)
        return x_train, y_train, x_test, y_test

    def build_model(self):
        self.model = tf.keras.Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(10, activation='sigmoid')
        ])
        self.model.compile(optimizer=Configuration.OPTIMIZER,
                           loss=Configuration.LOSS, metrics=[Configuration.METRICS])
        return self.model

    def train_model(self, model,train_features, train_label):
        self.history = model.fit(
            train_features, train_label, epochs=Configuration.EPOCHS, verbose=Configuration.VERBOSE, validation_split=0.2)
        return self.history

    def plot_loss(self, history):
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def predictions(self,model,x_test,y_test):
        self.y_pred = model.predict(x_test)
        self.number = 56
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        pd = PlotDigits.PlotDigits(self.number,self.y_pred[self.number],y_test)
        pd.plot_images(x_test)
        plt.subplot(1,2,1)
        pd.plot_value_array()
        plt.show()
        


def main():
    hwc = HandWritingClassification()
    (x_train, y_train), (x_test, y_test) = hwc.load_data()
    print("================The shape of the input data=====================")
    print(x_train.shape, x_test.shape)
    x_train, y_train, x_test, y_test=hwc.preprocess(x_train, y_train,x_test, y_test)
    print("==========Sample Preprocessed=======================")
    print(y_train[0],y_test[0])
    model = hwc.build_model()
    print("==================Model Summary====================")
    print(model.summary())
    print("=================Taining Model=====================")
    history =hwc.train_model(model,x_train,y_train)
    hwc.plot_loss(history)
    hwc.predictions(model,x_test,y_test)


if __name__ == '__main__':
    main()
