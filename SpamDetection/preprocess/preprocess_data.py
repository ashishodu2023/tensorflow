import tensorflow as tf
import os


class PreProcess:
    def __init__(self, logger, config_reader, texts, labels):
        self.num_records = None
        self.text_sequences = None
        self.logger = logger
        self.config_reader = config_reader
        self.texts = texts
        self.labels = labels
        self.cat_labels = None
        try:
            self.label_section = self.config_reader.config['LABEL']
            self.num_classes = self.label_section.get('num_classes')
            self.pad = self.label_section.get('dataset_name')
            self.shuffle = self.label_section.get('shuffle')
            self.batch_size = self.label_section.get('batch_size')
        except Exception as e:
            self.logger.exception(f'Error: Failed to get the values from config file.'
                                  f'Reason:Required key {e.args[0]} not found')

    def __tokenize_data(self):

        try:
            # Tokenize
            self.logger.info(f'***********Generating the tokens**********')
            tokenizer = tf.keras.preprocessing.text.Tokenizer()
            tokenizer.fit_on_texts(self.texts)
            text_sequences = tokenizer.texts_to_sequences(self.texts)
            self.num_records = len(text_sequences)
            max_seq_len = len(text_sequences[0])
            self.text_pad = tf.keras.preprocessing.sequence.pad_sequences(text_sequences,
                                                                          maxlen=max_seq_len, padding='post',
                                                                          truncating='post')
            # Labels
            self.logger.info(f'***********Generating the labels**********')
            self.cat_labels = tf.keras.utils.to_categorical(self.labels, num_classes=int(self.num_classes))
            self.logger.info(f'Sequence and Max length :{self.num_records},{max_seq_len}')
            # Vocab
            word2idx = tokenizer.word_index
            idx2word = {value: key for key, value in word2idx.items()}
            word2idx[self.pad] = 0
            idx2word[0] = self.pad
            self.vocab_size = len(word2idx)
            self.logger.info(f'Vocab size:{self.vocab_size}')
            return self.text_pad, self.cat_labels
        except Exception as e:
            self.logger.exception(f'Error: Error preprocessing the data.Reason.{e}')
            raise Exception(f'Error: Error preprocessing the data.Reason:{e}')

    def train_test_split(self):
        try:
            self.logger.info(f'***********Generating tensors**********')
            self.text_sequences, self.cat_labels = self.__tokenize_data()
            dataset = (tf.data.Dataset.from_tensor_slices((self.text_sequences, self.cat_labels))
                       .shuffle(int(self.shuffle)))
            test_size = self.num_records // 4
            val_size = (self.num_records - test_size) // 10
            test_dataset = dataset.take(test_size).batch(int(self.batch_size), drop_remainder=True)
            val_dataset = dataset.skip(test_size).take(val_size).batch(int(self.batch_size), drop_remainder=True)
            train_dataset = dataset.skip(test_size + val_size).batch(int(self.batch_size), drop_remainder=True)
            self.logger.info(f'***********Generated train validation and test datasets**********')
            self.logger.info(f'The length of train set:{len(train_dataset)}')
            self.logger.info(f'The length of validation set:{len(val_dataset)}')
            self.logger.info(f'The length of test set:{len(test_dataset)}')
            return train_dataset,val_dataset, test_dataset
        except Exception as e:
            self.logger.exception(f'Error: Failed to get train and test samples from dataset.'
                                  f'Reason:Required key {e.args[0]} not found')
