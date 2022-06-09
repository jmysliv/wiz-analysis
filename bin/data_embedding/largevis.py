import utils
import pandas as pd
import os


def save_txt(data, filename):
    with open(filename, 'w') as f:
        f.truncate(0)
        f.write(f'{str(data.shape[0])} {str(data.shape[1])}\n')
        for sample in data:
            for feature in sample:
                f.write(f'{str(feature)} ')
            f.write('\n')


def read_txt(filename):
    with open(filename, 'r+', encoding='utf-8') as f:
        data = f.read().splitlines(True)
    with open('tmp.txt', 'w') as f:
        f.truncate(0)
        f.writelines(data[1:])
    df = pd.read_csv('tmp.txt', sep=' ', dtype='float', header=None)
    return df


def embed(dataset, labels, name):
    save_txt(dataset, 'input.txt')
    os.system('./LargeVis/Linux/LargeVis -input input.txt -output output.txt')
    output = read_txt('output.txt').to_numpy()
    utils.save_plot_2d_scatter(output, labels, f"{name}_largevis.png")

    return output, labels
