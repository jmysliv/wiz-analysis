import pandas as pd
import numpy as np


def get_data(size):
    mnist = pd.read_csv('data/mnist.csv', names=([i for i in range(784)] + ['labels']))
    mnist_labels = np.array(mnist['labels'])
    mnist.pop('labels')
    mnist_values = np.array(mnist)
    return mnist_values[:size], mnist_labels[:size]
    