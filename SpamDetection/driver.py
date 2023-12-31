from utils.command_parser import CommandParser
from utils.config_reader import ConfigReader
from utils.logger_config import Logger
from data.get_sms_data import GetSmsData
from preprocess.preprocess_data import PreProcess
from model.model_train_test import SpamClassifierTrainer


class Driver:
    def __init__(self):
        self.model = None
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.labels = None
        self.texts = None
        self.logger = None
        self.args = None
        self.config_reader = None
        self.sms_data = None
        self.preprocess = None

    def parse_command_line_args(self):

        try:
            self.logger.info(f'Command line arguments parser')
            command_parser = CommandParser(self.logger)
            self.args = command_parser.parse_args()
            self.logger.info(f'Command line argument parsed:{self.args}')

        except Exception as e:
            self.logger.exception(f'Error: Error occurred while parsing the command line arguments. Reason.{e}')
            raise

    def main(self):
        try:
            self.logger = Logger.setup_logger()
            self.logger.info(f'Initializing logger in {__name__}')
            self.parse_command_line_args()

            self.config_reader = ConfigReader(self.logger)
            self.config_reader.read_config_file(self.args.config)
            if (self.args.data == 'get-data'
                    and self.args.split == 'split-data' and self.args.model == 'compile-model'):
                self.download_data()
                self.train_test_split()
                self.model_build_and_train()
        except Exception as e:
            self.logger.exception('An error occurred')
            raise

    def download_data(self):
        try:
            self.sms_data = GetSmsData(self.logger, self.config_reader)
            self.texts, self.labels = self.sms_data.download_and_read()
        except Exception as e:
            self.logger.exception(f'Error occurred at creating sms download object. Reason:{e}')
            raise

    def train_test_split(self):
        try:
            self.preprocess = PreProcess(self.logger, self.config_reader, self.texts, self.labels)
            self.train_dataset, self.val_dataset, self.test_dataset = self.preprocess.train_test_split()
        except Exception as e:
            self.logger.exception(f'Error occurred at splitting  the data. Reason:{e}')
            raise

    def model_build_and_train(self):
        try:
            self.preprocess = PreProcess(self.logger, self.config_reader, self.texts, self.labels)
            vocab_sz, e = self.preprocess.get_embedding_matrix()
            _, max_seq_len = self.preprocess.get_sequence()
            model_compile = SpamClassifierTrainer(self.logger, self.config_reader, vocab_sz, 300,
                                                  max_seq_len, 256, 3, 2,
                                                  'scratch', e)
            model_compile.build_model()
            self.model = model_compile.compile_model()
            model_compile.train_model(self.train_dataset, self.val_dataset)
            model_compile.test_model(self.test_dataset)
        except Exception as e:
            self.logger.exception(f'Error occurred at building the mode. Reason:{e}')
            raise


if __name__ == '__main__':
    Driver().main()
