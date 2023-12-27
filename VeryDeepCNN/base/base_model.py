class BaseModel(object):

    def __init__(self, config):
        self.config = config
        self.model = None

    # save function that saves the checkpoint in the path defined in the config file
    def save_model(self, checkpoint_path):
        if self.model is None:
            raise Exception('You have to build the model first.')

        print('Saving Model')
        self.model.save_weights(checkpoint_path)
        print('Model Saved')

    # load latest checkpoint from the experiment path defined in the config file
    def load_model(self, checkpoint_path):
        if self.model is None:
            raise Exception('You have to build the model first.')
        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print('Model Loaded')

    def build_model(self):
        raise NotImplementedError
