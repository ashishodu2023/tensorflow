from abc import ABC, abstractmethod
class BaseModel(ABC):

    @abstractmethod
    def load_data(self):
        pass
    
    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train_model(self, X_train, y_train, X_test, y_test):
        pass
