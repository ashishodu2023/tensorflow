from base.base_model import BaseModel
from keras.models import Model
from keras.layers import Input, Dense, MaxPooling2D, Dropout, Flatten, ZeroPadding2D, Convolution2D


class ConvCifarModel(BaseModel):

    def __init__(self, config):
        self.output_layer = None
        self.input_layer = None
        super(ConvCifarModel, self).__ini__(config)
        self.model = None
        self.build_model()

    def build_model(self):
        self.input_layer = Input(shape=(224, 224, 3))
        zero_1 = ZeroPadding2D((1, 1))(self.input_layer)
        conv_1 = Convolution2D(64, (3, 3), activation='relu')(zero_1)
        zero_2 = ZeroPadding2D((1, 1))(conv_1)
        conv_2 = Convolution2D(64, (3, 3), activation='relu')(zero_2)
        max_1 = MaxPooling2D((2, 2), strides=(2, 2))(conv_2)

        zero_3 = ZeroPadding2D((1, 1))(max_1)
        conv_3 = Convolution2D(64, (3, 3), activation='relu')(zero_3)
        zero_4 = ZeroPadding2D((1, 1))(conv_3)
        conv_4 = Convolution2D(64, (3, 3), activation='relu')(zero_4)
        max_2 = MaxPooling2D((2, 2), strides=(2, 2))(conv_4)
        flatten = Flatten()(max_2)
        dense_1 = Dense(4096, activation='relu')(flatten)
        drop_1 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu')(drop_1)
        drop_2 = Dropout(0.5)(dense_2)
        self.output_layer = Dense(1000, activation='softmax')(drop_2)

        self.model = Model(input=self.input_layer, output=self.output_layer)

        self.model.compile(
              loss='sparse_categorical_crossentropy',
              optimizer=self.config.model.optimizer,
              metrics=['accuracy'])
