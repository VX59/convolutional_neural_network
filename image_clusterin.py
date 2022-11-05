import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import math
import numpy as np
from preprocessor import input_pipeline
import random

classes = 8
width = 32

dataset = input_pipeline(width,classes)

convolution = keras.Sequential([
    tf.keras.layers.Conv2D(16, 3),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='valid'),
    tf.keras.layers.Conv2D(16, 3),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='valid'),
    tf.keras.layers.Conv2D(16, 3),
    tf.keras.layers.Flatten()
])

# create samples
samples = []
for i in range(100):
    sample = dataset.create_sample()
    sample = convolution(sample)
    samples.append(np.array(sample))
print(samples[0].shape)

domain = np.array(tf.range(samples[0].shape[1]))
samples = np.array(samples)

# select random centers
centroids = []
temp_samples = samples
for c in range(classes):
    rand = random.randint(0, len(temp_samples)-1)
    centroids.append(temp_samples[rand])
    np.delete(temp_samples, rand)

plt.scatter(domain, centroids[0][0], color='black', s=3)
plt.scatter(domain, samples[0][0], color='red', s=3)
plt.show()

new_centroids = []

def calc_distance(data, centers):
    diffs = []
    for i in range(len(data)):
        diffs.append(int(data[i] - centers[i]))

    reduced_diffs = np.sum(diffs)
    return reduced_diffs

def avg_points(cluster):
    print("cluster: " , np.array(cluster))
    if len(cluster) > 0:
        shape = len(cluster[0][0])
        reduced_cluster = np.array(tf.zeros(shape))    # same shape as each data point
        print("cluster size: ",  len(cluster))
        ratio = np.array(tf.fill(shape, len(cluster)))
        for c in cluster:   # each image
            reduced_cluster = np.add(c, reduced_cluster)

        avg = np.divide(reduced_cluster, ratio)
        return avg
    else:
        print('more samples required to evaluate, there are empty clusters...')
        exit(-1)

def compare_centroids(new, old):
    equal = False
    for i in range(len(new)):
        compare  = np.equal(new[i], old[i])
        if False not in compare: equal = True
        else: equal = False
    return equal

i = 0
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

    for cluster in clusters:
        avg = avg_points(cluster)
        #print("average: ", np.array(avg))
        #input()
        new_centroids.append(avg)

    plt.scatter(domain, new_centroids[0][0], color='black', s=3)
    plt.scatter(domain, samples[0][0], color='red', s=3)
    #plt.show()
    if compare_centroids(new_centroids, centroids): break
    else:
        i += 1 
        centroids = new_centroids
        new_centroids = []
    print("\n",'-'*50,"\n")

print("cycles: " ,i)