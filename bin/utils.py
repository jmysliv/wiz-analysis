import yaml
import matplotlib.pyplot as plt
import numpy as np

CONFS = None


def load_confs(confs_path='conf/conf.yaml'):
    global CONFS
    if CONFS is None:
        with open(confs_path, "r") as stream:
            try:
                CONFS = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    return CONFS


def get_conf(conf_name):
    """
    Get a configuration parameter by its name
    :param conf_name: Name of a configuration parameter
    :type conf_name: str
    :return: Value for that conf (no specific type information available)
    """
    return load_confs()[conf_name]


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

    plt.title("W")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'outputs/{filename}')