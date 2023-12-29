from utils.command_parser import CommandParser
from utils.config_reader import ConfigReader
from utils.logger_config import Logger


class Driver:
    def __init__(self):

        self.logger = None
        self.args = None
        self.config_reader = None

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
            if self.args.option == 'train-model':
                self.train_model()

        except Exception as e:
            self.logger.exception(f'An error occurred: {e}')
            raise


if __name__ == '__main__':
    Driver().main()
