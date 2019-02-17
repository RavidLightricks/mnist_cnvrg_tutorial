'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


from keras.utils.data_utils import get_file
import numpy as np


def load_data(path='mnist.npz'):
    """Loads the MNIST dataset.
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)
    
import argparse

parser = argparse.ArgumentParser(description='set input arguments')
parser.add_argument('--epochs', action="store",
                    dest='epochs', type=int, default=10)

parser.add_argument('--dropout', action="store",
                    dest='dropout', type=float, default=0.2)

parser.add_argument('--batch_size', action="store",
                    dest='batch_size', type=int, default=128)

parser.add_argument('--hidden', action="store",
                    dest='hidden', type=int, default=512)

parser.add_argument('--learning_rate', action="store",
                    dest='learning_rate', type=float, default=0.0001)
parser.add_argument('--dataset_path', action="store",
                    dest='dataset_path', type=str, default="/data/mnist_data/mnist.npz")

args = parser.parse_args()

epochs        = args.epochs
batch_size    = args.batch_size
hidden_nodes  = args.hidden
dropout       = args.dropout
learning_rate = args.learning_rate
dataset_path = args.dataset_path



num_classes = 10

print("cnvrg_tag_batch_size:", batch_size)
print("cnvrg_tag_epochs:", epochs)
print("cnvrg_tag_hidden_layers:", hidden_nodes)
print("cnvrg_tag_dropout:", dropout)
print("cnvrg_tag_learning_rate:", learning_rate)

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = load_data(dataset_path)

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(hidden_nodes, activation='relu', input_shape=(784,)))
model.add(Dropout(dropout))
model.add(Dense(hidden_nodes, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=2, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=2)

print('cnvrg_tag_TestLoss:', score[0])
print('cnvrg_tag_TestAccuracy:', score[1])

model.save_weights('model.weights')

print('cnvrg_tag_RMSE:', "0.24444")
print('cnvrg_tag_SMSE_ALL:', "0.341")
