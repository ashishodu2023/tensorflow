from BaseModel import BaseModel
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from tensorflow.keras.layers  import Dense,Normalization
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class FuelRegression(BaseModel):

    def __init__(self,epochs,optimizer,loss,verbose):
        self.EPOCHS = epochs
        self.model = None
        self.optimizer='adam'
        self.loss = 'mean_squared_error'
        self.verbose = verbose
        self.data_normalizer = None


    def load_data(self,url):
        self.columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model_year','origin','car name']
        self.data =  pd.read_csv(url,names=self.columns,na_values='?',comment='\t',
                   sep=' ',skipinitialspace=True,on_bad_lines='skip',encoding='UTF-8')
        self.data = self.data.drop(columns=['origin','car name'],axis=1)
        self.data = self.data.dropna()
        return self.data

    def train_test_split(self,data):
        self.train_ds = data.sample(frac=0.8,random_state=42)
        self.test_ds = data.drop(self.train_ds.index)
        return self.train_ds,self.test_ds

    def visualize_variables(self,train_ds):
        sns.pairplot(train_ds[['mpg','cylinders','displacement','horsepower','weight','acceleration','model_year']],diag_kind="hist")
        plt.show()
            

    def build_model(self,train_features):
        self.data_normalizer = Normalization(axis=1)
        self.data_normalizer.adapt(tf.convert_to_tensor(train_features,dtype='float32'))

        self.model = tf.keras.models.Sequential([
        self.data_normalizer,
        Dense(64,activation='relu',input_shape=(train_features.shape[1],),),
        Dense(32,activation='relu',input_shape=(train_features.shape[1],),),
        Dense(1,activation=None)
        ])
        print(self.model.summary())
        self.model.compile(optimizer=self.optimizer,loss=self.loss)

    def train_model(self,train_features,train_label):
        self.history = self.model.fit(train_features,train_label, epochs=self.EPOCHS,verbose=self.verbose,validation_split=0.2)
        return self.history

    def plot_loss(self,history):
        plt.plot(history.history['loss'],label='loss')
        plt.plot(history.history['val_loss'],label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Error [MPG]')
        plt.legend()
        plt.show()

    def predictions(self,test_features,test_label):
        self.y_pred = self.model.predict(test_features).flatten()
        ax = plt.axes(aspect='equal')
        plt.scatter(test_label,self.y_pred)
        plt.xlabel('True Value [MPG]')
        plt.ylabel('Predictions [MPG]')
        lims = [0,50]
        plt.xlim(lims)
        plt.ylim(lims)
        plt.plot(lims,lims)
        plt.show()
        return self.y_pred
    
    def plot_errors(self,y_pred,test_label):
        error = y_pred - test_label
        plt.hist(error, bins=30)
        plt.xlabel('Prediction Error [MPG]')
        plt.ylabel('Count')
        plt.show()



def main():
    URL ='./Regression/data/auto-mpg'
    flr = FuelRegression(200,'adam','mean_squared_error',1)
    data = flr.load_data(URL)
    print('=====================Fuel Price Dataset===============')
    print(data.head())
    data =data[~data.index.duplicated()]
    train_ds,test_ds=flr.train_test_split(data)
    flr.visualize_variables(train_ds)
    input_size = train_ds.shape[1]
    train_label = train_ds.pop('mpg')
    test_label = test_ds.pop('mpg')
    flr.build_model(train_ds)
    history=flr.train_model(train_ds,train_label)
    flr.plot_loss(history)
    y_pred=flr.predictions(test_ds,test_label)
    flr.plot_errors(y_pred,test_label)


if __name__=='__main__':
    main()

    


