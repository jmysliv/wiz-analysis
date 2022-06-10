import pandas as pd
import numpy as np

REUTERS_SIZE = 804409
def get_data(size = REUTERS_SIZE):
    reuters = pd.read_csv('data/reuters.csv', names=([i for i in range(30)] + ['labels']))
    reuters_labels = np.array(reuters['labels'])
    reuters.pop('labels')
    reuters_values = np.array(reuters)

    return reuters_values[:size], reuters_labels[:size]
