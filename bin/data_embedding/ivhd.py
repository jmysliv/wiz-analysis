import utils
import pandas as pd
import os

def save_data_as_csv(data, filename):
    with open(filename, 'w') as f:
        f.truncate(0)
        for i in range(1, data.shape[1] + 1):
            f.write(f'{str(i)}')
            if i != data.shape[1]:
                f.write(',')
        f.write('\n')
        for idx_sample, sample in enumerate(data):
            for idx, value in enumerate(sample):
                f.write(f'{str(value)}')
                if idx != len(sample) - 1:
                    f.write(',')
            if idx_sample != len(data) - 1:
                f.write('\n')


def save_target_as_csv(data, filename, with_shape=True):
    with open(filename, 'w') as f:
        f.truncate(0)
        if with_shape:
            f.write(f'{str(data.shape[0])} {str(data.shape[1])}\n')
        for idx, feature in enumerate(data):
            f.write(f'{str(feature)}')
            if idx != len(data) - 1:
                f.write('\n')


graph_names = {
    "MNIST": "mnist_cosine.bin",
    "20NG": '20ng_reduced_cosine.bin',
    "REUTERS": "reuters_reduced_cosine.bin"
}

arguments = {
    "MNIST": '4500 2 1 1 0 0 0 force-directed',
    "20NG": '6000 3 1 1 0 0 0 force-directed',
    "REUTERS": '6000 3 1 1 0 0 0 force-directed'
}

input_x = 'input_X.csv'
input_y = 'input_Y.csv'
output = 'output.csv'


def embed(dataset, labels, name):
    core_path = './viskit/viskit_offline'
    graph_name = graph_names[name]
    argument = arguments[name]
    save_data_as_csv(dataset, f'{core_path}/input/{input_x}')
    save_target_as_csv(labels, f'{core_path}/input/{input_y}', False)
    command = f'{core_path}/viskit_offline {core_path}/input/{input_x} ' \
              f'{core_path}/input/{input_y} {core_path}/graphs/{graph_name} {core_path}/output/{output} {argument}'
    os.system(command)
    res = pd.read_csv(f'{core_path}/output/{output}').iloc[:, :2].to_numpy()
    new_labels = pd.read_csv(f'{core_path}/output/{output}').iloc[:, 2].to_numpy()
    utils.save_plot_2d_scatter(res, new_labels, f"{name}_ivhd.png")

    return res, new_labels
