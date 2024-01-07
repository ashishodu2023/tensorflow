from utils.command_parser import CommandParser
from utils.config_reader import ConfigReader
from utils.logger_config import Logger
from data.get_book_data import BookData


class Driver:
    def __init__(self):
        self.idx2char = None
        self.char2idx = None
        self.book_data = None
        self.model = None
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.labels = None
        self.logger = None
        self.args = None
        self.config_reader = None
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
            if self.args.data == 'get-data':
                self.download_data()
        except Exception as e:
            self.logger.exception('An error occurred')
            raise

    def download_data(self):
        try:
            self.book_data = BookData(self.logger, self.config_reader)
            self.char2idx, self.idx2char = self.book_data.get_char2idx()
            return self.char2idx, self.idx2char
        except Exception as e:
            self.logger.exception(f'Error occurred at creating sms download object. Reason:{e}')
            raise


if __name__ == '__main__':
    Driver().main()
