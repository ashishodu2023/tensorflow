class BasePreProcess(object):
    def __init__(self,data, config):
        self.config = config
        self.data = data

    def get_normalize_data(self):
        raise NotImplementedError

    def get_normalized_test_data(self):
        raise  NotImplementedError
