import sklearn.datasets as data
import matplotlib.pyplot as plt
from math import sqrt
import random
from numpy.random import choice


X, clusters = data._samples_generator.make_blobs(n_samples=1000, n_features=2, cluster_std=1, random_state=10)

# x = []
# y = []
# for value in X:
#     x.append(value[0])
#     y.append(value[1])

# plt.scatter(x, y)

# for pair in X:
#     plt.scatter(pair[0], pair[1])

# plt.show()

class Cluster():
    def __init__(self, p):
        self.values = [p]
        self.centroid = p

    def update_centroid(self):
        x, y = 0, 0
        for value in self.values:
            x += value[0]
            y += value[1]
        x = x / len(self.values)
        y = y / len(self.values)
        self.centroid = (x, y)

    def scatter_cluster(self, plot, colour):
        for pair in self.values:
            plot.scatter(pair[0], pair[1], color=colour)   


def get_distance(p1, p2):
    return sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))

def calculate_wc(clusters):
    WC = 0
    for cluster in clusters:
        for point in cluster.values:
            WC += get_distance(cluster.centroid, point) ** 2
    return WC

def calculate_bc(clusters):
    BC = 0
    for i in range(len(clusters) - 1):
        for j in range(i + 1, len(clusters) - 1):
            BC += get_distance(clusters[i].centroid, clusters[j].centroid) ** 2
    return BC

def calculate_CH_index(clusters, number_of_points=1000):
    BC = calculate_bc(clusters)
    WC = calculate_wc(clusters)
    chIndex = (BC / (len(clusters) - 1)) / (WC / (number_of_points - len(clusters)))
    return chIndex


def k_means(data, k, plot):
    colors = ["red", "blue", "green", "yellow", "purple", "brown", "black", "orange"]
    clusters = []
    for i in range(k):
        clusters.append(Cluster(data[random.randint(0,len(data)-1)]))

    convergence_value = 0.1
    difference = float('inf')
    while difference > convergence_value:
        for cluster in clusters:
            cluster.values = []
        for point in data:
            min = float('inf')
            closest_cluster = None
            for cluster in clusters:
                d = get_distance(point, cluster.centroid)
                if d < min:
                    min = d
                    closest_cluster = cluster
            closest_cluster.values.append(point)
        centroids = [c.centroid for c in clusters]
        closest_cluster.update_centroid()
        differences = [get_distance(clusters[i].centroid, centroids[i]) for i in range(len(centroids))]
        difference = max(differences)
    
    print(calculate_CH_index(clusters=clusters))

    i = 0
    for cluster in clusters:
        cluster.scatter_cluster(plot, colors[i])
        i += 1
    plot.show()


# calculate_wc(3)

k_means(X, 3, plt)