import numpy as np
import matplotlib.pyplot as plt


class PlotDigits:

    def __init__(self,number,prediction_array,y_true):
        self.color = None
        self.number = number
        self.prediction_array = prediction_array
        self.y_true = y_true
        

    def plot_images(self,images):
        self.y_true, self.images = self.y_true[self.number], images[self.number]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.images, cmap=plt.cm.binary)

        self.y_pred = np.argmax(self.prediction_array)
        if self.y_pred == self.y_true:
            self.color = 'blue'
        else:
            self.color = 'red'

        plt.xlabel("Pred {} Conf: {:2.0f}% True ({})".format(self.y_pred, 100*np.max(self.prediction_array), self.y_true, color=self.color))

    def plot_value_array(self):
        self.y_true = self.y_true[self.number]
        plt.grid(False)
        plt.xticks([10])
        plt.yticks([])
        thisplot = plt.bar(range(10),self.prediction_array,color='#777777')
        plt.ylim([0,1])
        self.y_pred = np.argmax(self.prediction_array)
        thisplot[self.y_pred].set_color('red')
        thisplot[self.y_true].set_color('blue')

