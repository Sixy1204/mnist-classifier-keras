# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt 
% matplotlib
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255.   # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# build your neural net
model = Sequential([
    Dense(128, input_dim=784),
    Activation('relu'),
    Dropout(0.5),
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(10),
    Activation('softmax'),
])

#define your optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# add metrics to get more results you want to see
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# train the model
model.fit(X_train, y_train, epochs=100, batch_size=128)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)

##Prediction##
X_test_0 = X_test[101,:].reshape(1,784)
y_test_0 = y_test[101,:]
plt.imshow(X_test_0.reshape([28,28]))

pred = model.predict(X_test_0[:])

#real data label
print('Label of testing sample', np.argmax(y_test_0))
#predicted data
print('Output of the softmax layer', pred[0]) #probability
print('Network prediction:', np.argmax([pred[0]])) #label
