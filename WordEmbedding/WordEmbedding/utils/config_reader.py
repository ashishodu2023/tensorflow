import os
import configparser


class ConfigReader:

    def __init__(self, logger):

        try:
            self.logger = logger
            self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
            self.config_path = f'{os.path.dirname(os.path.dirname(__file__))}/dependencies/config'
        except Exception as e:
            self.logger.exception(f'Error: Failed to initialize the ConfigReader object. Reason.{e}')
            raise Exception(f'Error: Failed to initialize the ConfigReader object.Reason:{e}')

    def read_config_file(self, config_file):

        try:
            self.logger.info(f'Loading config file: {self.config_path}{config_file}')
            self.config.read(f'{self.config_path}{config_file}')
            self.logger.info(f'{self.config_path}{config_file} loaded successfully')
        except Exception as e:
            self.logger.exception(f'Error: Failed to read the configuration file. Reason.{e}')
            raise Exception(f'Error: Failed to read the configuration file.Reason:{e}')
