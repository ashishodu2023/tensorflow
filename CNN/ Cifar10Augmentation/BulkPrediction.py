from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt


# Load Model

class BulkPrediction:

    def __init__(self, model_arch, model_weights):
        self.classes = None
        self.predictions = None
        self.image = None
        self.images = None
        self.model_arch = model_arch
        self.model_weights = model_weights
        self.model = None
        self.labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
        self.R = 5
        self.C = 5

    # Load Data
    def load_data(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        return (x_train, y_train), (x_test, y_test)

    # Load Model
    def load_model(self):
        print("===============Load Model===================")
        self.model = model_from_json(open(self.model_arch).read())
        self.model.load_weights(self.model_weights)
        print(self.model.summary())

    # Train
    def train(self):
        print("===============Compiling Model===================")
        self.model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])
        print("===============Model Compiled====================")

    def predict(self, x_test, y_test):
        print("===============Predictions===================")
        predictions = self.model.predict(x_test)
        Y_pred_classes = np.argmax(predictions, axis=1)
        Y_true = np.argmax(y_test, axis=1)
        fig, axes = plt.subplots(self.R, self.C, figsize=(12, 12))
        axes = axes.ravel()

        for i in np.arange(0, self.R * self.C):
            axes[i].imshow(x_test[i])
            axes[i].set_title("True: %s \nPredict: %s" % (self.labels[Y_true[i]], self.labels[Y_pred_classes[i]]))
            axes[i].axis('off')
            plt.subplots_adjust(wspace=1)
        plt.show()


def main():
    bp = BulkPrediction('model.json', 'model.h5')
    (x_train, y_train), (x_test, y_test) = bp.load_data()
    bp.load_model()
    bp.train()
    bp.predict(x_test, y_test)


if __name__ == '__main__':
    main()
