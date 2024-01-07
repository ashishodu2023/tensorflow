import numpy as np
from model.model_classifier import SpamClassifierCNN
from sklearn.metrics import accuracy_score, confusion_matrix


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
        self.labels = []
        self.predictions = []

        try:
            self.logger.info('Inside SpamClassifier Trainer..')
            self.label_section = self.config_reader.config['MODEL_DEFINITION']
            self.conv_filters = self.label_section.get('conv_num_filters')
            self.kernel_size = self.label_section.get('conv_kernal_size')
            self.optimizer = self.label_section.get('optimizer')
            self.loss = self.label_section.get('loss')
            self.model_metrics = self.label_section.get('model_metrics')
            self.num_epochs = self.label_section.get('num_epochs')
            self.class_weight = self.label_section.get('class_weight')
            self.model = SpamClassifierCNN(self.logger, self.config_reader, self.vocab_sz, self.embed_sz,
                                           self.input_length,
                                           self.num_filters, self.kernel_sz, self.output_sz, self.run_mode,
                                           self.embedding_weights)

        except Exception as e:
            self.logger.exception(f'Error: Failed to create the model instance.Reason: {e}')
            raise

    def build_model(self):
        try:
            self.logger.info('Inside Build Model..')
            self.logger.info(self.input_length)
            self.model.build(input_shape=(None, self.input_length))
            self.logger.info('***************Model Summary***************')
            self.logger.info(self.model.summary())
        except Exception as e:
            self.logger.exception(f'Error: Failed to build the model.Reason: {e}')
            raise

    def compile_model(self):
        try:
            self.logger.info('Inside Compile Model..')
            self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.model_metrics])
            self.logger.info('Model Compiled..')
        except Exception as e:
            self.logger.exception(f'Error: Failed to compile.Reason: {e}')
            raise

    def train_model(self, train_dataset, val_dataset):
        try:
            self.logger.info('Start model training..')
            self.model.fit(train_dataset, epochs=int(self.num_epochs),
                           validation_data=val_dataset, class_weight=self.class_weight)
            self.logger.info('Training Completed..')
            return self.model
        except Exception as e:
            self.logger.exception(f'Error: Failed to train the model.Reason: {e}')
            raise

    def test_model(self, test_dataset):
        try:
            self.logger.info('Start model testing..')
            for x_test, y_test in test_dataset:
                y_test_ = self.model.predict_on_batch(x_test)
                y_test = np.argmax(y_test, axis=1)
                y_test_ = np.argmax(y_test_, axis=1)
                self.labels.extend(y_test.tolist())
                self.predictions.extend(y_test_.tolist())
            self.logger.info(f'Test Accuracy:{accuracy_score(self.labels, self.predictions)}')
            self.logger.info('Confusion Matrix')
            self.logger.info(confusion_matrix(self.labels, self.predictions))
        except Exception as e:
            self.logger.exception(f'Error: Failed to predict the model.Reason: {e}')
            raise
