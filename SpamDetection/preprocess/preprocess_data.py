import numpy as np
import tensorflow as tf
import os
import gensim.downloader as api

class PreProcess:
    def __init__(self, logger, config_reader, texts, labels):
        self.num_records = None
        self.text_sequences = None
        self.logger = logger
        self.config_reader = config_reader
        self.texts = texts
        self.labels = labels
        self.cat_labels = None
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
        try:
            self.label_section = self.config_reader.config['LABEL']
            self.num_classes = self.label_section.get('num_classes')
            self.pad = self.label_section.get('dataset_name')
            self.shuffle = self.label_section.get('shuffle')
            self.batch_size = self.label_section.get('batch_size')

            self.label_embed = self.config_reader.config['EMBEDDING']
            self.embedding_dim = self.label_embed.get('embedding_dim')
            self.embedding_model = self.label_embed.get('embedding_model')
            self.data_dir = self.label_embed.get('data_dir')
        except Exception as e:
            self.logger.exception(f'Error: Failed to get the values from config file.'
                                  f'Reason:Required key {e.args[0]} not found')

    def __tokenize_data(self):

        try:
            # Tokenize
            self.logger.info(f'Inside tokensize data..')
            self.tokenizer.fit_on_texts(self.texts)
            text_sequences = self.tokenizer.texts_to_sequences(self.texts)
            self.num_records = len(text_sequences)
            max_seq_len = len(text_sequences[0])
            self.text_pad = tf.keras.preprocessing.sequence.pad_sequences(text_sequences,
                                                                          maxlen=max_seq_len, padding='post',
                                                                          truncating='post')
            # Labels
            self.logger.info(f'***********Generating the labels**********')
            self.cat_labels = tf.keras.utils.to_categorical(self.labels, num_classes=int(self.num_classes))
            self.logger.info(f'Sequence and Max length :{self.num_records},{max_seq_len}')
            return self.text_pad, self.cat_labels
        except Exception as e:
            self.logger.exception(f'Error: Error preprocessing the data.Reason.{e}')
            raise Exception(f'Error: Error preprocessing the data.Reason:{e}')

    def __get_word2idx_data(self):
        try:
            # Vocab
            self.logger.info(f'Inside word2idx ..')
            self.tokenizer.fit_on_texts(self.texts)
            word2idx = self.tokenizer.word_index
            idx2word = {value: key for key, value in word2idx.items()}
            word2idx[self.pad] = 0
            idx2word[0] = self.pad
            self.vocab_size = len(word2idx)
            self.logger.info(f'Vocab size:{self.vocab_size}')
            return word2idx
        except Exception as e:
            self.logger.exception(f'Error: Error creating word2idx data.Reason.{e}')
            raise Exception(f'Error: Error preprocessing the data.Reason:{e}')

    def train_test_split(self):
        try:
            self.logger.info(f'Inside train_test_split ..')
            self.text_sequences, self.cat_labels = self.__tokenize_data()
            dataset = (tf.data.Dataset.from_tensor_slices((self.text_sequences, self.cat_labels))
                       .shuffle(int(self.shuffle)))
            test_size = self.num_records // 4
            val_size = (self.num_records - test_size) // 10
            test_dataset = dataset.take(test_size).batch(int(self.batch_size), drop_remainder=True)
            val_dataset = dataset.skip(test_size).take(val_size).batch(int(self.batch_size), drop_remainder=True)
            train_dataset = dataset.skip(test_size + val_size).batch(int(self.batch_size), drop_remainder=True)
            X_train = list(map(lambda x: x[0], train_dataset))
            y_train = list(map(lambda x: x[1], train_dataset))
            X_val = list(map(lambda x: x[0], val_dataset))
            y_val = list(map(lambda x: x[1], val_dataset))
            X_test = list(map(lambda x: x[0], test_dataset))
            y_test = list(map(lambda x: x[1], test_dataset))

            self.logger.info(f'***********Generated train validation and test datasets**********')
            self.logger.info(f'The shape of train set:{np.asarray(X_train).shape,np.asarray(y_train).shape}')
            self.logger.info(f'The shape of validation set:{np.asarray(X_val).shape,np.asarray(y_val).shape}')
            self.logger.info(f'The shape of test set:{np.asarray(X_test).shape,np.asarray(y_test).shape}')
            return train_dataset,val_dataset, test_dataset
        except Exception as e:
            self.logger.exception(f'Error: Failed to get train and test samples from dataset.'
                                  f'Reason:Required key {e.args[0]} not found')


    def get_sequence(self):
        try:
            self.logger.info(f'Inside get_sequence..')
            self.tokenizer.fit_on_texts(self.texts)
            text_sequences = self.tokenizer.texts_to_sequences(self.texts)
            self.num_records = len(text_sequences)
            max_seq_len = len(text_sequences[0])
            text_pad = tf.keras.preprocessing.sequence.pad_sequences(text_sequences,
                                                                            maxlen=max_seq_len, padding='post',
                                                                            truncating='post')
            return text_pad,max_seq_len
        except Exception as e:
            self.logger.exception(f'Error:Failed to get the sequence lengths.'
                              f'Reason:Required key {e.args[0]} not found')
    def get_embedding_matrix(self):
        self.logger.info(f'Inside embedding matrix..')
        embedding_file= os.path.join(self.data_dir,'E.npy')
        word2idx=self.__get_word2idx_data()
        self.vocab_sz = len(word2idx)
        sequences,_ =self.get_sequence()
        try:
            if os.path.exists(embedding_file):
                E = np.load(embedding_file)
            else:
                E = np.zeros((self.vocab_sz,int(self.embedding_dim)))
                word_vectors = api.load(self.embedding_model)
                for word,idx in word2idx.items():
                    try:
                        E[idx]=word_vectors.word_vec(word)
                    except KeyError:
                        pass
                np.save(embedding_file,E)
            return  self.vocab_sz,E
        except Exception as e:
            self.logger.exception(f'Error:Failed to create the embedding matrix.'
                              f'Reason:Required key {e.args[0]} not found')




