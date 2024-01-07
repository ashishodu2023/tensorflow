import tensorflow as tf
import re


class BookData(object):

    def __init__(self, logger, config_reader):

        self.logger = logger
        self.config_reader = config_reader
        self.texts = []

        try:
            self.logger.info(f'Inside BookData initializer..')
            self.data_section = self.config_reader.config['DATA']
            self.url1 = self.data_section.get('url1')
            self.url2 = self.data_section.get('url2')
            self.newline = self.data_section.get('newline')
            self.pattern = self.data_section.get('pattern')
            self.byte_marker = self.data_section.get('byte_marker')

            if self.url1 is None and self.url2 is None:
                self.logger.exception(f'The Url1 and Url2 cannot be empty')

        except Exception as e:
            self.logger.exception(f'Error: Failed to get the values from config file.'
                                  f'Reason:Required key {e.args[0]} not found')

    def __clean_data(self, text):
        try:
            self.logger.info(f'Inside clean data..')
            text = text.replace(self.byte_marker, '')
            text = text.replace(self.newline, ' ')
            text = re.sub(self.pattern, ' ', text)
            return text
        except Exception as e:
            self.logger.exception(f'Error: Failed to clean the data.')

    def get_char2idx(self):
        try:
            self.logger.info(f'Inside get_char2idx..')
            self.texts = self.__download_and_read()
            vocab = sorted(set(self.texts))
            self.logger.info(f'Vocab Size: {len(vocab)}')
            char2idx = {c: i for i, c in enumerate(vocab)}
            idx2char = {i: c for c, i in char2idx.items()}
            return char2idx, idx2char
        except Exception as e:
            self.logger.exception(f'Error: Failed to clean the data.')

    def __download_and_read(self):
        try:
            self.logger.info(f'Inside download and read data..')
            for idx, url in enumerate([self.url1, self.url2]):
                p = tf.keras.utils.get_file('ex1-{:d}.txt'.format(idx), url, cache_dir='.')
                text = open(p, 'r', encoding="utf8").read()
                text = self.__clean_data(text)
                self.texts.extend(text)
            return self.texts
        except Exception as e:
            self.logger.exception(f'Error: Failed to download the data from urls')
