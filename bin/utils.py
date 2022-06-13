import matplotlib.pyplot as plt
import numpy as np


def save_plot_2d_scatter(X, labels, filename):
    fig, plot = plt.subplots()
    fig.set_size_inches(12, 12)
    plt.prism()

    for i in np.unique(labels):
        elements = (labels == i)

        dim0 = X[elements, 0]
        dim1 = X[elements, 1]
        label = f"Class: {i}"
        plot.scatter(dim0, dim1, label=label)
    
    plot.set_xticks(())
    plot.set_yticks(())

    plt.tight_layout()
    plt.legend()
    plt.savefig(f'outputs/{filename}')