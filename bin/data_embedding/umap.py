import umap
import utils

def embed(dataset, labels):
    output = umap.UMAP().fit_transform(dataset)
    utils.save_plot_2d_scatter(output, labels, "mnist_umap.png")