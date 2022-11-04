import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import math
import numpy as np
from preprocessor import input_pipeline
import random

classes = 4
width = 32

dataset = input_pipeline(width,classes)

convolution = keras.Sequential([
    tf.keras.layers.Conv2D(16, 3),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='valid'),
    tf.keras.layers.Flatten()
])

# create samples
samples = []
for i in range(10):
    sample = dataset.create_sample()

    sample = convolution(sample)
    samples.append(np.array(sample))

domain = np.array(tf.range(3600))
samples = np.array(samples)

# select random centers
centroids = []
temp_samples = samples
for c in range(classes):
    rand = random.randint(0, len(temp_samples)-1)
    centroids.append(temp_samples[rand])
    np.delete(temp_samples, rand)

plt.scatter(domain, centroids[0][0], color='black', s=2)
plt.scatter(domain, samples[0][0], color='red', s=2)
#plt.show()

new_centroids = []


def calc_distance(data, centers):
    diffs = []
    print(data, centers)
    for i in range(len(data)):
        diffs.append(int(data[i] - centers[i]))

    reduced_diffs = np.sum(diffs)
    return reduced_diffs

def avg_points(cluster):
    print(cluster)
    input()
    if len(cluster) > 0:
        shape = cluster[0].shape[0]
        reduced_cluster = np.array(tf.zeros([shape,shape]))    # same shape as each data point
        ratio = np.array(tf.fill(shape, len(cluster)))

        for point in cluster:   # each image
            np.add(point, reduced_cluster)
    
        avg = np.divide(reduced_cluster, ratio)
        return avg
    else:
        return None

def compare_centroids(new, old):
    for i in range(len(new)):   # each centroid
        print(new[i])
    
    return False


while(centroids != new_centroids):
    clusters = []
    for i in range(classes): clusters.append([])

    for s in samples:
        distances = []
        for c in centroids:
            distance = calc_distance(s[0], c[0])
            distances.append(abs(distance))

        cluster = np.argmin(distances)
        clusters[cluster].append(s)

    print(clusters)

    for cluster in clusters:
        avg = avg_points(cluster)
        print(avg)
        input()
        new_centroids.append(avg)

    if compare_centroids(new_centroids, centroids): break
    else: 
        centroids = new_centroids
        new_centroids = []

