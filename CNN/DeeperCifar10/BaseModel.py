from abc import ABC, abstractmethod
class BaseModel(ABC):

    @abstractmethod
    def load_data(self):
        pass
    
    @abstractmethod
    def build_model(self,input_shape):
        pass

    @abstractmethod
    def train_model(self, X_train, y_train,x_test, y_test):
        pass
