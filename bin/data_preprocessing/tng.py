import pandas as pd
import numpy as np

TNG_SIZE = 9000


def get_data(size = TNG_SIZE):
    tng = pd.read_csv('data/tng.csv', names=([i for i in range(5000)] + ['labels']))
    tng_labels = np.array(tng['labels'])
    tng.pop('labels')
    tng_values = np.array(tng)
    return tng_values[:size], tng_labels[:size]
    