import utils
from data_preprocessing import mnist
from data_embedding import tsne, umap



if __name__ == '__main__':
    # example usage
    size = utils.get_conf('size')

    # mnist
    mnist_values, mnist_labels = mnist.visualize_mnist(size)
    tsne.embed(mnist_values, mnist_labels)
    umap.embed(mnist_values, mnist_labels)

