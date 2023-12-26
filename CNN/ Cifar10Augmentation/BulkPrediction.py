import numpy as np
from matplotlib.pyplot import imread
from skimage.transform import resize
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD


# Load Model

class BulkPrediction:

    def __init__(self, model_arch, model_weights):
        self.image = None
        self.images = None
        self.model_arch = model_arch
        self.model_weights = model_weights
        self.img_name = ['cat-standing.jpg', 'dog.jpg']
        self.model = None

    # Load Model
    def load_model(self):
        print("===============Load Model===================")
        self.model = model_from_json(open(self.model_arch).read())
        self.model.load_weights(self.model_weights)
        print(self.model.summary())

    # Load Images
    def load_images(self):
        print("===============Load Test Images===================")
        self.images = [np.transpose(resize(imread(image), (32, 32)),
                                    (2, 0, 1)).astype('float32')
                       for image in self.img_name
                       ]
        # self.images = [np.transpose(Image.fromarray(obj=img, mode='F').resize(size=(32, 32), resample=Image.BICUBIC))
        #                for img in self.img_name]
        self.images = np.array(self.images) / 255.0
        print(self.images)

    # Train
    def train(self):
        print("===============Compiling Model===================")
        self.model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

    def predict(self):
        print("===============Predictions===================")
        self.predictions = self.model.predict(self.images)
        self.classes = np.argmax(self.predictions, axis=1)
        print(self.classes)


def main():
    bp = BulkPrediction('model.json', 'model.h5')
    bp.load_model()
    bp.load_model()
    bp.train()
    bp.predict()


if __name__ == '__main__':
    main()
