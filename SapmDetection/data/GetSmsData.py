import tensorflow as tf
import os

# TODO: URL should come from config
class GetSmsData:
    def __init__(self, logger, url):
        try:
            self.logger = logger
            self.label = []
            self.text = []
            if url is None:
                self.logger(f'The URL cannot be empty')
            else:
                self.url = url
        except Exception as e:
            self.logger.exception(f'Error: Failed to initialize the Logger object.Reason.{e}')
            raise Exception(f'Error: Failed to initialize the Logger object.Reason:{e}')

    def download_and_read(self):
        try:
            local_file = self.url.split('/')[-1]
            p = tf.keras.utils.get_file(local_file, self.url, extract=True, cache_dir='.')
            local_file = os.path.join('datasets', 'SMSSpamCollection')

            try:
                with open(local_file, 'r') as file_obj:
                    for line in file_obj:
                        label, text = line.strip().split('\t')
                        self.label.append(1 if label == 'spam' else 0)
                        self.text.append(text)
                return self.text, self.label
            except Exception as e:
                self.logger.exception(f'Error: Open the file.Reason.{e}')
                raise Exception(f'Error:  Open the file.Reason:{e}')
        except Exception as e:
            self.logger.exception(f'Error: Downloading the data from Url.Reason.{e}')
            raise Exception(f'Error: Downloading the data from Url.Reason:{e}')
