import pandas as pd
import numpy as np
from bin.utils import root_path


def get_data(size):
    reuters = pd.read_csv(root_path + '/data/reuters.csv', names=([i for i in range(30)] + ['labels']))
    reuters_labels = np.array(reuters['labels'])
    reuters.pop('labels')
    reuters_values = np.array(reuters)
    return reuters_values[:size], reuters_labels[:size]
