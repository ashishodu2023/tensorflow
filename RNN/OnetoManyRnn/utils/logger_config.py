import logging
import os
import socket
import datetime


class Logger:

    @staticmethod
    def setup_logger():
        try:
            # Create logger
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)

            # Create log formatter
            log_format = '%(asctime)s :%(name)s : %(lineno)d :%(levelname)s :%(message)s'
            formatter = logging.Formatter(log_format)

            # Create the console handler and set the formatter
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            # Add console handler to the logger
            logger.addHandler(console_handler)

            # Create the file handler and set the formatter
            log_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
            os.makedirs(log_directory, exist_ok=True)
            current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            host_name = socket.gethostname()
            log_file = os.path.join(log_directory, f'wd_{current_time}_{host_name}.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)

            # Add the file handler to the logger
            logger.addHandler(file_handler)

            return logger
        except Exception as e:
            print(f'Error occurred while setting up the logger: {e}')
            raise
