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


def embed(dataset, labels, graph_name, name):
    core_path = './viskit/viskit_offline'
    save_data_as_csv(dataset, 'input_X.csv')
    save_target_as_csv(labels, 'input_Y.csv', False)
    command = f'.${core_path}/viskit_offline input_X.csv output_Y.csv graphs/{graph_name} output.csv'
    os.system(command)
    output = pd.read_csv(f'output.csv').iloc[:, :2].to_numpy()
    utils.save_plot_2d_scatter(output, labels, f"{name}_ivhd.png")

    return output
