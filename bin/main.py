import utils
from data_preprocessing import mnist
from data_embedding import tsne, umap, largevis
from metrics.local_score import LocalMetric
from metrics.time_score import TimeScore
from metrics.clustering_score import ClusterScore
from metrics.spearman_score import SpearmanScore



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
        time_metric = TimeScore(dataset_name)
        cluster_metric = ClusterScore(dataset_name)
        spearman_metric = SpearmanScore(dataset_name)

        for method, method_name in zip(methods, method_names):
            print(f"{method_name} embedding...")
            time_metric.start_measurement()
            output = method.embed(values, labels, dataset_name)

            # add results to metrics
            time_metric.stop_measurement(method_name)
            local_metric.calculate_knn_gain_and_dr_quality(
                X_lds=output,
                X_hds=values,
                labels=labels,
                method_name=method_name
            )
            cluster_metric.calculate_cluster_score(output, labels, method_name)
            spearman_metric.calculate_score(values, output, method_name)

        # Metrics comparison
        local_metric.visualize()
        time_metric.visualize()
        cluster_metric.visualize()
        spearman_metric.visualize()