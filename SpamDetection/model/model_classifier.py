import tensorflow as tf


class SpamClassifierCNN(tf.keras.Model):

    def __init__(self, logger, config_reader, vocab_sz, embed_sz, input_length,
                 num_filters, kernel_sz, output_sz, run_mode, embedding_weights, **kwargs):
        super(SpamClassifierCNN, self).__init__(**kwargs)
        self.logger = logger
        self.config_reader = config_reader
        self.vocab_sz = vocab_sz
        self.embed_sz = embed_sz
        self.input_length = input_length
        self.num_filters = num_filters
        self.kernel_sz = kernel_sz
        self.output_sz = output_sz
        self.run_mode = run_mode
        self.embedding_weights = embedding_weights

        try:
            self.logger.info('Inside the SpamClassifierCNN..')
            self.label_section = self.config_reader.config['MODEL']
            self.scratch = self.label_section.get('scratch')
            self.vectorizer = self.label_section.get('vectorizer')
            self.relu = self.label_section.get('relu')
            self.dropout = self.label_section.get('dropout')
            self.activation = self.label_section.get('activation')

            if self.run_mode == self.scratch:
                self.embedding = tf.keras.layers.Embedding(self.vocab_sz, self.embed_sz, input_length=self.input_length,
                                                           trainable=True)
            elif self.run_mode == self.vectorizer:
                self.embedding = tf.keras.layers.Embedding(self.vocab_sz, self.embed_sz, input_length=self.input_length,
                                                           trainable=False, weights=[self.embedding_weights])
            else:
                self.embedding = tf.keras.layers.Embedding(self.vocab_sz, self.embed_sz, input_length=self.input_length,
                                                           trainable=True, weights=[self.embedding_weights])

            self.conv = tf.keras.layers.Conv1D(filters=self.num_filters, kernel_size=self.kernel_sz,
                                               activation=self.relu)
            self.dropout = tf.keras.layers.SpatialDropout1D(float(self.dropout))
            self.pool = tf.keras.layers.GlobalMaxPooling1D()
            self.dense = tf.keras.layers.Dense(self.output_sz, activation=self.activation)
        except Exception as e:
            self.logger.exception(f'Error: Failed to create the model definition.Reason: {e}')
            raise

    def call(self, x, training=None, mask=None):
        try:
            x = self.embedding(x)
            x = self.conv(x)
            x = self.dropout(x)
            x = self.pool(x)
            x = self.dense(x)
            return x
        except Exception as e:
            self.logger.exception(f'Error: Failed to create the model layers.Reason: {e}')
            raise
