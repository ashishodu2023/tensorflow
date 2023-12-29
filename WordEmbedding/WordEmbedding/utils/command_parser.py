import argparse


class CommandParser(object):

    def __init__(self, logger):
        self.logger = logger
        self.parser = argparse.ArgumentParser(
            description='Word Embeddings models', usage='python3 driver.py -o <option> -c <config_file_name>',
            epilog='Use --help to see more options'
        )

        self.parser.add_argument('-o', '--option', help='Options -train model', required=True)
        self.parser.add_argument('-c', '--config', help='Name of the config file', required=True)
        self.args = self.parser.parse_args()

    def parse_args(self):

        try:
            return self.args
        except Exception as e:
            self.logger.exception(f'Error: Failed to parse the command line arguments. Reason.{e}')
            raise Exception(f'Error: Failed to parse the command line arguments.Reason:{e}')
