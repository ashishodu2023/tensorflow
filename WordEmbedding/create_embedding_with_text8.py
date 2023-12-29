import errno
import os

import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.models import Word2Vec


class Text8Embedding:
    def __init__(self):
        self.word_vectors = None
        self.model = None
        self.model_name = 'text8-word2vec.bin'

    def __get_data(self):
        return api.load('text8')

    def create_model(self):
        self.model = Word2Vec(self.__get_data())

    def save_model(self):
        model_dir = 'WordEmbedding/data'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, self.model_name)
        self.model.save(model_path)

    def load_model(self):
        model_path = os.path.join('WordEmbedding/data', self.model_name)
        if os.path.isfile(model_path):
            self.model = KeyedVectors.load(model_path)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_path)

    def explore_embeddings(self):
        self.word_vectors = self.model.wv
        return list(self.word_vectors.index_to_key)

    @staticmethod
    def show_embeddings(words, num_words=10):
        print(words[:num_words])
        assert 'king' in words

    def main(self):
        text8 = Text8Embedding()
        text8.create_model()
        text8.save_model()
        text8.load_model()
        words = text8.explore_embeddings()
        text8.show_embeddings(words, num_words=10)


if __name__ == '__main__':
    Text8Embedding().main()
