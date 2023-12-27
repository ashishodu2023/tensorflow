from base.base_model import BaseModel
from keras.models import Model
from keras.layers import Input, Dense, MaxPooling2D, Dropout, Flatten, ZeroPadding2D, Convolution2D


class ConvCifarModel(BaseModel):

    def __init__(self, config):
        self.output_layer = None
        self.input_layer = None
        super(ConvCifarModel, self).__init__(config)
        self.model = None
        self.build_model()

    def build_model(self):
        self.input_layer = Input(shape=(32, 32, 3))
        self.zero_1 = ZeroPadding2D((1, 1))(self.input_layer)
        self.conv_1 = Convolution2D(64, (3, 3), activation='relu')(self.zero_1)
        self.zero_2 = ZeroPadding2D((1, 1))(self.conv_1)
        self.conv_2 = Convolution2D(64, (3, 3), activation='relu')(self.zero_2)
        self.max_1 = MaxPooling2D((2, 2), strides=(2, 2))(self.conv_2)

        self.zero_3 = ZeroPadding2D((1, 1))(self.max_1)
        self.conv_3 = Convolution2D(64, (3, 3), activation='relu')(self.zero_3)
        self.zero_4 = ZeroPadding2D((1, 1))(self.conv_3)
        self.conv_4 = Convolution2D(64, (3, 3), activation='relu')(self.zero_4)
        self.max_2 = MaxPooling2D((2, 2), strides=(2, 2))(self.conv_4)
        self.flatten = Flatten()(self.max_2)
        self.dense_1 = Dense(4096, activation='relu')(self.flatten)
        self.drop_1 = Dropout(0.5)(self.dense_1)
        self.dense_2 = Dense(4096, activation='relu')(self.drop_1)
        self.drop_2 = Dropout(0.5)(self.dense_2)
        self.output_layer = Dense(1000, activation='softmax')(self.drop_2)

        self.model = Model(inputs=self.input_layer, outputs=self.output_layer)

        self.model.compile(
              loss='categorical_crossentropy',
              optimizer=self.config.model.optimizer,
              metrics=['accuracy'])
