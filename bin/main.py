from data_preprocessing import mnist, tng, reuters
from data_embedding import tsne, umap, largevis, ivhd
from metrics.local_score import LocalMetric
from metrics.time_score import TimeScore
from metrics.clustering_score import ClusterScore


methods = [ivhd, tsne, umap, largevis]
method_names = ["IVHD", "TSNE", "UMAP", "LARGEVIS"]
datasets = [mnist, tng, reuters]
dataset_names = ["MNIST", "20NG", "REUTERS"]


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
            local_metric.calculate_knn_gain_and_dr_quality(
                X_lds=output,
                X_hds=values,
                labels=new_labels,
                method_name=method_name
            )
            cluster_metric.calculate_cluster_score(output, new_labels, method_name)

        # Metrics comparison
        local_metric.visualize(dataset_name)
        time_metric.visualize()
        cluster_metric.visualize()
