import matplotlib.pyplot as plt
import math
import numpy as np

data = [(1,2),(1,4),(2,3),(4,1),(8,5),(9,6)]
init_centers = [(1,3), (3,2)]

def get_dist(data, center):
    return math.sqrt(math.pow(data[0] - center[0], 2) + math.pow(data[1] - center[1], 2))

def average_points(points):
    sum = [0,0]
    for x,y  in points:
        sum[0] += x
        sum[1] += y
    
    return (sum[0] / len(points), sum[1] / len(points))

new_centers = []

plt.scatter([x[0] for x in data],
            [y[1] for y in data])
plt.scatter([x[0] for x in init_centers],
            [y[1] for y in init_centers], color='red')
plt.show()

while(init_centers != new_centers):

    clusters = []
    for i in range(len(init_centers)):
        clusters.append([])

    for d in data:
        distances = []
        for c in init_centers:
            distances.append(get_dist(d, c))
        cluster = np.argmin(distances)
        clusters[cluster].append(d)

    print("\n", "-"*30, "\n")
    print("clusters: ", np.array(clusters))

    new_centers = []

    for i in range(len(clusters)):
        avg = average_points(clusters[i])
        new_centers.append(avg)

    print("centers: ", new_centers)

    plt.scatter([x[0] for x in data],
            [y[1] for y in data])
    plt.scatter([x[0] for x in new_centers],
            [y[1] for y in new_centers], color='red')

    for c in clusters:
        for data_point in c:
            center = new_centers[clusters.index(c)]
            endpoints = [data_point, center]

            plt.plot([x[0] for x in endpoints],
                     [y[1] for y in endpoints], color='green')
    
    plt.show()

    if new_centers == init_centers: break
    else: 
        init_centers = new_centers
        new_centers = []