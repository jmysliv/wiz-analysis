import pandas as pd
import numpy as np

MNIST_SIZE = 70000
def get_data(size = MNIST_SIZE):
    mnist = pd.read_csv('data/mnist.csv', names=([i for i in range(784)] + ['labels']))
    mnist_labels = np.array(mnist['labels'])
    mnist.pop('labels')
    mnist_values = np.array(mnist)
    return mnist_values[:size], mnist_labels[:size]
    