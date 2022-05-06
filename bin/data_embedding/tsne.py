from sklearn.manifold import TSNE
import utils

def embed(dataset, labels):
    output = TSNE(n_components=2, metric='euclidean', perplexity=30).fit_transform(dataset)
    utils.save_plot_2d_scatter(output, labels, "mnist_tsne.png")