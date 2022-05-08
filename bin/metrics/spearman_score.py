from scipy import stats
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import numpy as np



class SpearmanScore():
    def __init__(self, dataset):
        self.dataset = dataset
        self.scores = []
        self.labels = []

    def calculate_distances(self, data):
        distances = distance_matrix(data, data)
        distances_flatten = np.triu(distances).flatten()
        return distances_flatten[distances_flatten != 0]

    def calculate_score(self, original_data, embedded_data, method_name):
        original_distances = self.calculate_distances(original_data)
        embedded_distances = self.calculate_distances(embedded_data)
        score = stats.spearmanr(original_distances, embedded_distances)
        self.scores.append(score.correlation)
        self.labels.append(method_name)

    def visualize(self):
        fig = plt.figure(figsize=(16, 12))
        plt.title = 'Spearman Correlation Score'
        plt.ylabel('Score')
        plt.bar(self.labels, self.scores)
        plt.savefig(f'outputs/{self.dataset}_spearman_score.png')
        plt.close()