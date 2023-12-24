from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing

class BaseModel(ABC):
    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train_model(self, X_train, y_train, X_test, y_test):
        pass

class TextClassifier(BaseModel):
    def __init__(self, max_len, n_words, dim_embedding, epochs, batch_size):
        self.MAX_LEN = max_len
        self.N_WORDS = n_words
        self.DIM_EMBEDDING = dim_embedding
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size
        self.model = self.build_model()

    def load_data(self):
        (X_train, y_train), (X_test, y_test) = datasets.imdb.load_data(num_words=self.N_WORDS)

        X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=self.MAX_LEN)
        X_test = preprocessing.sequence.pad_sequences(X_test, maxlen=self.MAX_LEN)

        return (X_train, y_train), (X_test, y_test)

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Embedding(self.N_WORDS, self.DIM_EMBEDDING, input_length=self.MAX_LEN))
        model.add(layers.Dropout(0.3))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def train_model(self, X_train, y_train, X_test, y_test):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print(self.model.summary())
        score = self.model.fit(X_train, y_train, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE,
                               validation_data=(X_test, y_test))

        score = self.model.evaluate(X_test, y_test, batch_size=self.BATCH_SIZE)
        print("\nTest Score:", score[0])
        print("\nTest Accuracy:", score[1])

def main():
    text_classifier = TextClassifier(max_len=200, n_words=10000, dim_embedding=200, epochs=20, batch_size=500)
    (X_train, y_train), (X_test, y_test) = text_classifier.load_data()
    text_classifier.train_model(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()
