import tensorflow as tf 
import numpy as np 
from tensorflow import keras

# Network and training parameters
EPOCHS = 200
BATCH_SIZE  = 128
VERBOSE = 1
NB_CLASSES = 10
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2
RESHAPED = 784
DROP_OUT = 0.3

mnist = keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(60000,RESHAPED).astype('float32')
x_test = x_test.reshape(10000,RESHAPED).astype('float32')

# Normalize the data
x_train /=255
x_test /=255

#print(x_train.shape[0],'train_samples')
#print(x_test.shape[0],'test_samples')

y_train = tf.keras.utils.to_categorical(y_train,NB_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test,NB_CLASSES)

# Build model
model = tf.keras.models.Sequential()
model.add(
    keras.layers.Dense(
        N_HIDDEN,
        input_shape = (RESHAPED,),
        name ='dense_layer_1',
        activation='relu'
    )
)
model.add(keras.layers.Dropout(DROP_OUT))
model.add(
    keras.layers.Dense(
        N_HIDDEN,
        input_shape = (RESHAPED,),
        name ='dense_layer_2',
        activation='relu'
    )
)
model.add(keras.layers.Dropout(DROP_OUT))
model.add(
    keras.layers.Dense(
        NB_CLASSES,
        input_shape = (RESHAPED,),
        name ='dense_layer_3',
        activation='softmax'
    )
)

#Compiling the model
#model.compile(optimizer='SGD',loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='RMSProp',loss='categorical_crossentropy', metrics=['accuracy']) #0.9771
model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy']) #0.9807

print(model.summary())

#Traning the model
model.fit(x_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=VERBOSE,validation_split=VALIDATION_SPLIT)


#Evaluate the model
test_loss,test_acc = model.evaluate(x_test,y_test)
print('Test Accuracy:', test_acc)