import utils
from data_preprocessing import mnist
from data_embedding import tsne, umap, largevis
from metrics.local_score import LocalMetric


methods = [tsne, umap, largevis]
method_names = ["TSNE", "UMAP", "LARGEVIS"]
datasets = [mnist]
dataset_names = ["MNIST"]


if __name__ == '__main__':
    # configure size in conf/conf.yaml if you want smaller datasets
    size = utils.get_conf('size')

    for dataset, dataset_name in zip(datasets, dataset_names):
        values, labels = dataset.get_data(size)
        print(f"--------{dataset_name}--------")

        # initialize metrics
        local_metric = LocalMetric()

        for method, method_name in zip(methods, method_names):
            print(f"{method_name} embedding...")
            output = method.embed(values, labels, dataset_name)

            # add results to metrics
            local_metric.calculate_knn_gain_and_dr_quality(
                X_lds=output,
                X_hds=values,
                labels=labels,
                method_name=method_name
            )

        # Metrics comparison
        local_metric.visualize()
