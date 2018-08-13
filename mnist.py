# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt 
#%matplotlib
from sklearn.cross_validation import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils


# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X = X_train.reshape(X_train.shape[0], -1) / 255.   # normalize
y = np_utils.to_categorical(y_train, num_classes=10)
#split X and y to train data set and validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=0)
#test set
X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # normalize
y_test = np_utils.to_categorical(y_test, num_classes=10)

#build your neural net
model = Sequential([
    #avoid intermediate layers with fewer than final outputs dimensional units
    Dense(512, input_dim=784), 
    Activation('relu'),
    Dropout(0.5), #avoid overfitting
    Dense(512, input_dim=784), 
    Activation('relu'),
    Dropout(0.5), #avoid overfitting
    Dense(512, input_dim=784), 
    Activation('relu'),
    Dropout(0.5), #avoid overfitting
    Dense(10),
    Activation('softmax'),
])

#define your optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

#We add metrics to get more results you want to see
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# train the model
mf=model.fit(X_train, y_train, epochs=13 , batch_size=500, validation_data=(X_val, y_val))

#Plotting the training and validation loss
mf_dict=mf.history
train_loss=mf_dict['loss']
val_loss=mf_dict['val_loss']
epochs = np.arange(1, len(mf_dict['acc']) + 1)
plt.figure('loss plot')
plt.plot(epochs, train_loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

#Plotting the training and validation accuracy
plt.figure('accuracy')
train_acc = mf_dict['acc']
val_acc = mf_dict['val_acc']
plt.plot(epochs, train_acc, 'ro', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

#test the model
print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)
print('test loss: ', loss)
print('test accuracy: ', accuracy)

##Prediction##
X_test_0 = X_test[101,:].reshape(1,784)
y_test_0 = y_test[101,:]
plt.figure('Prediction Picture')
plt.imshow(X_test_0.reshape([28,28]))

pred = model.predict(X_test_0[:])

#real data label
print('Label of testing sample', np.argmax(y_test_0))
#predicted data
print('Output of the softmax layer', pred[0]) #probability
print('Network prediction:', np.argmax([pred[0]])) #label
