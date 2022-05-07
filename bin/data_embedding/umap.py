import umap
import utils


def embed(dataset, labels, name):
    output = umap.UMAP().fit_transform(dataset)
    utils.save_plot_2d_scatter(output, labels, f"{name}_umap.png")

    return output
