from data_preprocessing import mnist, tng, reuters
from data_embedding import tsne, umap, largevis, ivhd
from metrics.local_score import LocalMetric
from metrics.time_score import TimeScore
from metrics.clustering_score import ClusterScore

methods = [ivhd, tsne, umap, largevis]
method_names = ["IVHD", "TSNE", "UMAP", "LARGEVIS"]
datasets = [mnist, reuters, tng]
dataset_names = ["MNIST", "REUTERS", "20NG"]


def get_half_dataset(dataset):
    n = dataset.shape[0]
    return dataset[:n//2]


if __name__ == '__main__':

    for dataset, dataset_name in zip(datasets, dataset_names):
        values, labels = dataset.get_data()
        print(f"--------{dataset_name}--------")

        # initialize metrics
        local_metric = LocalMetric()
        time_metric = TimeScore(dataset_name)
        cluster_metric = ClusterScore(dataset_name)

        for method, method_name in zip(methods, method_names):
            print(f"{method_name} embedding...")
            time_metric.start_measurement()
            output, new_labels = method.embed(values, labels, dataset_name)

            # add results to metrics
            time_metric.stop_measurement(method_name)

            embedded = get_half_dataset(output)
            original = get_half_dataset(values)
            y = get_half_dataset(new_labels)
            local_metric.calculate_knn_gain_and_dr_quality(
                X_lds=embedded,
                X_hds=original,
                labels=y,
                method_name=method_name
            )
            cluster_metric.calculate_cluster_score(embedded, y, method_name)

        # Metrics comparison
        local_metric.visualize(dataset_name)
        time_metric.visualize()
        cluster_metric.visualize()
