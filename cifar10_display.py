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
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

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












