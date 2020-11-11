<<<<<<< HEAD
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QLineEdit, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore
from matplotlib imprt pyplot as plt
import sys
import glob
import os
=======
>>>>>>> 6f80e15b28127e7669c713710785958fc0c6cc08
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
import random
import torch
import torchvision
import torchvision.transforms as transforms
from keras.utils import np_utils, plot_model
<<<<<<< HEAD
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import SGD

##### Window #####
app = QApplication(sys.argv)
win = QMainWindow()
win.setGeometry(550, 550, 1000, 500)
win.setWindowTitle('VGG16')

##### Q5 #####
#Label for Q1
label = QtWidgets.QLabel(win)
label.setText("1.Show Train Images")
label.adjustSize()

=======
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
>>>>>>> 6f80e15b28127e7669c713710785958fc0c6cc08

cifar10 = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_images = train_images/255
test_images = test_images/255
class_names = ('airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
#train_labels = np_utils.to_categorical(train_labels, class_names)
#test_labels = np_utils.to_categorical(test_labels, class_names)

print(train_images.shape)
print(train_labels.shape)

print(test_images.shape)
print(test_labels.shape)

class_names = ('airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
plt.figure(figsize  = (32, 32))

for i in range(10):
    x = np.random.choice(range(len(train_labels)))
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[x], cmap = plt.cm.binary)
    plt.xlabel(class_names[train_labels[x].item()])
plt.show()

train_labels = keras.utils.to_categorical(train_labels,10)
test_labels = keras.utils.to_categorical(test_labels,10)

batch_size = 64
num_classes = 10
epochs = 100

model = keras.Sequential()

<<<<<<< HEAD
weight_decay = 0.00005

model = keras.Sequential()
#Block 1
model.add(keras.layers.Conv2D(64,  kernel_size=(3,3),
                              #strides=(2,2),
                              activation='relu',
                              input_shape=(32,32,3),
                              padding='same'))
model.add(keras.layers.Conv2D(64,  kernel_size=(3,3),
                              #strides=(2,2),
                              activation='relu',
                              input_shape=(32,32,3),
                              padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
#Block 2
model.add(keras.layers.Conv2D(128, kernel_size=(3,3),
                              #strides=(2,2),
                              activation='relu',
                              input_shape=(32,32,3),
                              padding='same'))
model.add(keras.layers.Conv2D(128, kernel_size=(3,3),
                              #strides=(2,2),
                              activation='relu',
                              input_shape=(32,32,3),
                              padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
#Block 3
model.add(keras.layers.Conv2D(256, kernel_size=(3,3),
                              #strides=(2,2),
                              activation='relu',
                              input_shape=(32,32,3),
                              padding='same'))
model.add(keras.layers.Conv2D(256, kernel_size=(3,3),
                              #strides=(1,1),
                              activation='relu',
                              input_shape=(32,32,3),
                              padding='same'))
model.add(keras.layers.Conv2D(256, kernel_size=(3,3),
                              #strides=(1,1),
                              activation='relu',
                              input_shape=(32,32,3),
                              padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
#Block 4
model.add(keras.layers.Conv2D(512, kernel_size=(3,3),
                              #strides=(1,1),
                              activation='relu',
                              input_shape=(32,32,3),
                              padding='same'))
model.add(keras.layers.Conv2D(512, kernel_size=(3,3),
                              #strides=(1,1),
                              activation='relu',
                              input_shape=(32,32,3),
                              padding='same'))
model.add(keras.layers.Conv2D(512, kernel_size=(3,3),
                            #   strides=(1,1),
                              activation='relu',
                              input_shape=(32,32,3),
                              padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
#Block 5
model.add(keras.layers.Conv2D(512, kernel_size=(3,3),
                            #   strides=(1,1),
                              activation='relu',
                              input_shape=(32,32,3),
                              padding='same'))
model.add(keras.layers.Conv2D(512, kernel_size=(3,3),
                            #   strides=(1,1),
                              activation='relu',
                              input_shape=(32,32,3),
                              padding='same'))
model.add(keras.layers.Conv2D(512, kernel_size=(3,3),
                            #   strides=(1,1),
                              activation='relu',
                              input_shape=(32,32,3),
                              padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu',kernel_regularizer=keras.regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(512, activation='relu',kernel_regularizer=keras.regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))





model.summary()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
history = model.fit(train_images,train_labels batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_images,test_labels))

test_loss, test_acc = model.evaluate(test_image,test_label)
print('Test Accuracy',test_acc)

=======
>>>>>>> 6f80e15b28127e7669c713710785958fc0c6cc08











