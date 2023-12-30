import tensorflow as tf
import os


# TODO: URL should come from config
class GetSmsData:
    def __init__(self, logger, config_reader):
        self.logger = logger
        self.config_reader = config_reader
        try:
            self.data_section = self.config_reader.config['DATA']
            self.data_url = self.data_section.get('url')
            self.dataset_name = self.data_section.get('dataset_name')
            if self.data_url is None:
                self.logger.exception(f'The URL cannot be empty')
        except Exception as e:
            self.logger.exception(f'Error: Failed to get the values from config file.'
                                  f'Reason:Required key {e.args[0]} not found')
            exit(1)
        self.labels = []
        self.texts = []

    def download_and_read(self):
        try:
            zip_file = self.data_url.split('/')[-1]
            p = tf.keras.utils.get_file(zip_file, self.data_url, extract=True, cache_dir='.')
            local_file = os.path.join('datasets', self.dataset_name)
            try:
                with open(local_file, 'r') as file_obj:
                    for line in file_obj:
                        label, text = line.strip().split('\t')
                        self.labels.append(1 if label == 'spam' else 0)
                        self.texts.append(text)
                return self.texts, self.labels
                #print(self.texts[1], self.labels[1])
            except Exception as e:
                self.logger.exception(f'Error: Open the file.Reason.{e}')
                raise Exception(f'Error:  Open the file.Reason:{e}')
        except Exception as e:
            self.logger.exception(f'Error: Downloading the data from Url.Reason.{e}')
            raise Exception(f'Error: Downloading the data from Url.Reason:{e}')
