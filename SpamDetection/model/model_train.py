from model_classifier import SpamClassifierCNN


class SpamClassifierTrainer(SpamClassifierCNN):
    def __init__(self, logger, config_reader, vocab_sz, embed_sz, input_length,
                 num_filters, kernel_sz, output_sz, run_mode, embedding_weights):
        super(SpamClassifierTrainer, self).__init__(logger, config_reader, vocab_sz, embed_sz, input_length,
                                                    num_filters, kernel_sz, output_sz, run_mode, embedding_weights)
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
            self.logger.info('Inside SpamClassifier Trainer..')
            self.label_section = self.config_reader.config['MODEL_DEFINITION']
            self.conv_filters = self.label_section.get('conv_num_filters')
            self.kernel_size = self.label_section.get('conv_kernal_size')
            self.optimizer = self.label_section.get('optimizer')
            self.loss = self.label_section.get('loss')
            self.metrics = self.label_section.get('metrics')
            self.model = SpamClassifierCNN(self.logger, self.config_reader, self.vocab_sz, self.embed_sz,
                                           self.input_length,
                                           self.num_filters, self.kernel_sz, self.output_sz, self.run_mode,
                                           self.embedding_weights)

        except Exception as e:
            self.logger.exception(f'Error: Failed to create the model instance.Reason: {e}')
            raise

    def build_model(self, max_seqlen):
        self.model.build(input_shape=(None, max_seqlen))
        self.logger.info('***************Model Summary***************')
        self.logger.info(self.model.summary())

    def compile_model(self):
        self.logger.info('Inside Compile Model..')
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.metrics])
        return self.model
