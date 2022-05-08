from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt


class ClusterScore():
    def __init__(self, dataset, coverage = 0.5):
        self.dataset = dataset
        self.coverage = coverage
        self.scores = []
        self.labels = []

    def calculate_cluster_score(self, data, labels, method_name):
        unique_labels = np.unique(labels)
        # calculate different classes mean distance
        k_means = KMeans(n_clusters=unique_labels.shape[0]).fit(data).cluster_centers_
        distances = distance_matrix(k_means, k_means)
        distances = MinMaxScaler().fit_transform(distances)
        size = distances.shape[0]
        x = np.sum(distances)/(size*(size-1))
        
        # calculate same class mean distance
        means = []
        for label in unique_labels:
            label_points = data[labels == label]
            random_idxs_size = int(label_points.shape[0]*self.coverage)
            random_idxs = np.random.randint(label_points.shape[0], size=random_idxs_size)
            label_points = label_points[random_idxs, :]
            distances = distance_matrix(label_points, label_points)
            distances = MinMaxScaler().fit_transform(distances)
            mean = np.sum(distances)/(random_idxs_size*(random_idxs_size-1))
            means.append(mean)
        
        y = np.mean(means)

        # calculate score
        score = y/x
        self.scores.append(score)
        self.labels.append(method_name)

    def visualize(self):
        fig = plt.figure(figsize=(16, 12))
        plt.title = 'Clustering Score'
        plt.ylabel('Score')
        plt.bar(self.labels, self.scores)
        plt.savefig(f'outputs/{self.dataset}_cluster_score.png')
        plt.close()
