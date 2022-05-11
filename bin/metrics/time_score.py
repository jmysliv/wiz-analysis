import time
import matplotlib.pyplot as plt


class TimeScore():
    def __init__(self, dataset):
        self.dataset = dataset
        self.start_time = None
        self.times = []
        self.labels = []

    def start_measurement(self):
        self.start_time = time.time()

    def stop_measurement(self, method_name):
        self.times.append(time.time() - self.start_time)
        self.labels.append(method_name)

    def visualize(self):
        fig = plt.figure(figsize=(16, 12))
        plt.title = 'Time Score'
        plt.ylabel('Time [s]')
        plt.bar(self.labels, self.times)
        plt.savefig(f'outputs/{self.dataset}_time_score.png')
        plt.close()