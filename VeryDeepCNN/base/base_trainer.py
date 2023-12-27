
class BaseTrainer(object):

    def __init__(self,model,data,config):
        self.model = model
        self.data = data
        self.config = config

    def trainer(self):
        raise NotImplementedError
