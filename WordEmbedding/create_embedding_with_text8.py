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
        model_dir = 'data'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, self.model_name)
        self.model.save(model_path)

    def load_model(self):
        model_path = os.path.join('data', self.model_name)
        if os.path.isfile(model_path):
            self.model = KeyedVectors.load(model_path)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_path)

    def explore_embeddings(self):
        self.word_vectors = self.model.wv
        return list(self.word_vectors.index_to_key)

    @staticmethod
    def show_embeddings(words, num_words=10):
        print('\n*************Word Embeddings****************')
        print(words[:num_words])
        assert 'king' in words

    def get_model_vector(self):
        return self.model.wv

    @staticmethod
    def print_most_similar(word_conf_pairs, k):
        print('\n*************Similar Words****************')
        for i, (word, conf) in enumerate(word_conf_pairs):
            print(f'{conf :.3f} {word}')
            if i >= k - 1:
                break
        if k < len(word_conf_pairs):
            print('...')

    def main(self):
        text8 = Text8Embedding()
        text8.create_model()
        text8.save_model()
        text8.load_model()
        words = text8.explore_embeddings()
        word_vector = text8.get_model_vector()
        text8.show_embeddings(words, num_words=10)
        text8.print_most_similar(word_vector.most_similar('king'), 5)


if __name__ == '__main__':
    Text8Embedding().main()
