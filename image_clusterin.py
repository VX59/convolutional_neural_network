# messing around

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import math
import numpy as np
from preprocessor import input_pipeline
import random
import glob
from tqdm import tqdm
import imageio

classes = 4
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
for i in range(400):
    sample = dataset.create_sample()
    sample = convolution(sample)
    samples.append(sample)
    
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
plt.scatter(domain, centroids[1][0], color='blue', s=3)
plt.scatter(domain, centroids[2][0], color='red', s=3)
plt.scatter(domain, centroids[3][0], color='green', s=3)
plt.savefig("images/000.png")
#plt.show()
plt.close()

new_centroids = []

def calc_distance(data, centers):
    diffs = []
    for i in range(len(data)):
        term = math.pow(int(data[i] - centers[i]), 2)
        diffs.append(term)

    sum = np.sum(diffs)
    return math.sqrt(sum)    

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

a = 0
while(centroids != new_centroids):
    clusters = []
    for i in range(classes): clusters.append([])

    for s in samples:
        distances = []
        for c in centroids:
            distance = calc_distance(s[0], c[0])
            distances.append(distance)

        cluster = np.argmin(distances)
        clusters[cluster].append(s)

    for cluster in clusters:
        avg = avg_points(cluster)
        #print("average: ", np.array(avg))
        #input()
        new_centroids.append(avg)
    plt.scatter(domain, new_centroids[0][0], color='black', s=3)
    plt.scatter(domain, new_centroids[1][0], color='blue', s=3)
    plt.scatter(domain, new_centroids[2][0], color='red', s=3)
    plt.scatter(domain, new_centroids[3][0], color='green', s=3)
    print(a)
    plt.savefig(f"images/{a+1:03d}.png")
    plt.close()

    if compare_centroids(new_centroids, centroids): break
    else:
        centroids = new_centroids
        new_centroids = []
    a += 1

    print("\n",'-'*50,"\n")

print("cycles: " ,a)

def create_gif(path_to_img, name_gif):
    filenames = glob.glob(path_to_img)
    filenames = sorted(filenames)
    images = []
    for filename in tqdm(filenames):
        images.append(imageio.imread(filename))
    
    kargs = {"duration": 0.25}
    imageio.mimsave(name_gif, images, "GIF", **kargs)

create_gif("images/*.png", "clustering.gif")