# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:25:08 2020

@author: 19191600
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

'''
Task 1
'''
# Data Generation
area_pts = np.random.randint(0, 100, (30, 2))

# Plotting the data
plt.figure(dpi=150)
plt.scatter(x = area_pts[:, 0], y = area_pts[:, 1])

plt.xlabel('x-values')
plt.ylabel('y-values')
plt.title("Random Points")
plt.show()

# Initialising the algorithm
kmeans_estimator = KMeans(n_clusters=3, random_state=0)
k_clusters = kmeans_estimator.fit(area_pts) # train the algorithm

print(k_clusters.labels_)
print(k_clusters.cluster_centers_)
print(k_clusters.inertia_)

pointLabels = k_clusters.labels_
clusterCenters = k_clusters.cluster_centers_

# Plotting the data
plt.figure(dpi=150)
plt.scatter(x =area_pts[:, 0], y= area_pts[:, 1], 
            c=pointLabels,)
plt.scatter(x =clusterCenters[:,0], y= clusterCenters[:,1], 
            marker="x", c="r")
plt.title("Random Points with Cluster Centers")
plt.show()


'''
Exercise 1
'''
area_pts2 = np.random.randint(0, 100, (500, 2))

kmeans_estimator2 = KMeans(n_clusters=5, random_state=0) 
k_clusters = kmeans_estimator2.fit(area_pts2)

clusterCenters = k_clusters.cluster_centers_
pointsLabels = k_clusters.labels_

# Plotting the data
plt.figure(dpi=150)
plt.scatter(x=area_pts2[:, 0], y= area_pts2[:, 1], 
            c=pointsLabels, cmap=plt.cm.Set1)

plt.scatter(clusterCenters[:,0], clusterCenters[:,1], 
            marker="x", c="b")
plt.title("Exercise 1: Random Points with Cluster Centers")
plt.xlim(-1, 101)
plt.ylim(-1, 101)
plt.xlabel('x')
plt.ylabel('y')
plt.show()



'''
Task 3
'''
# Data Generation
rect_pts_x = np.random.randint(1, 15, (150, 1))
rect_pts_y = np.random.randint(1, 80, (150, 1))
rect_pts1 = np.concatenate((rect_pts_x, rect_pts_y), axis=1)

rect_pts_x = np.random.randint(31, 45, (150, 1))
rect_pts_y = np.random.randint(81, 160, (150, 1))
rect_pts2 = np.concatenate((rect_pts_x, rect_pts_y), axis=1)

rect_pts_x = np.random.randint(61, 75, (150, 1))
rect_pts_y = np.random.randint(1, 80, (150, 1))
rect_pts3 = np.concatenate((rect_pts_x, rect_pts_y), axis=1)

rect_pts_x = np.random.randint(91, 105, (150, 1))
rect_pts_y = np.random.randint(81, 160, (150, 1))
rect_pts4 = np.concatenate((rect_pts_x, rect_pts_y), axis=1)


sample_pts1 = np.concatenate((rect_pts1, rect_pts2, 
                              rect_pts3, rect_pts4), axis=0)

# Plotting the data
plt.figure(dpi=150)
plt.scatter(x= sample_pts1[:, 0], y=sample_pts1[:, 1])
plt.xlabel('x')
plt.ylabel('y')
# plt.xlim(-2, 162)
# plt.ylim(-2, 162)
plt.title("Random Cluster points")
plt.show()

# Initialising the algorithm
kmeans_estimator3 = KMeans(n_clusters=4, random_state=0)
k_clusters = kmeans_estimator3.fit(sample_pts1)

clusterCenters = k_clusters.cluster_centers_
pointsLabels = k_clusters.labels_

# Plotting the data
plt.figure(dpi=150)
plt.scatter(x=sample_pts1[:, 0], y= sample_pts1[:, 1], 
            c=pointsLabels, cmap=plt.cm.Set1)
plt.scatter(clusterCenters[:,0], clusterCenters[:,1], marker="x")
plt.xlabel('x-values')
plt.ylabel('y-values')
plt.show()

# Data Generation
rect_pts_x = np.random.randint(1, 15, (150, 1))
rect_pts_y = np.random.randint(1, 120, (150, 1))
rect_pts1 = np.concatenate((rect_pts_x, rect_pts_y), axis=1)

rect_pts_x = np.random.randint(31, 45, (150, 1))
rect_pts_y = np.random.randint(41, 160, (150, 1))
rect_pts2 = np.concatenate((rect_pts_x, rect_pts_y), axis=1)

rect_pts_x = np.random.randint(61, 75, (150, 1))
rect_pts_y = np.random.randint(1, 120, (150, 1))
rect_pts3 = np.concatenate((rect_pts_x, rect_pts_y), axis=1)

rect_pts_x = np.random.randint(91, 105, (150, 1))
rect_pts_y = np.random.randint(41, 160, (150, 1))
rect_pts4 = np.concatenate((rect_pts_x, rect_pts_y), axis=1)

sample_pts2 = np.concatenate((rect_pts1, rect_pts2, 
                              rect_pts3, rect_pts4), axis=0)

# Plotting the data
plt.figure(dpi=150)
plt.scatter(x= sample_pts2[:, 0], y=sample_pts2[:, 1])
plt.xlim(0, 106)
plt.ylim(0, 161)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Random Cluster points")
plt.show()

# Initialising the algorithm
kmeans_estimator4 = KMeans(n_clusters=4, random_state=0)
k_clusters = kmeans_estimator4.fit(sample_pts2)
clusterCenters = k_clusters.cluster_centers_

# Plotting the data
plt.figure(dpi=150)
plt.scatter(sample_pts2[:, 0], sample_pts2[:, 1], 
            c=k_clusters.labels_, cmap=plt.cm.Set1)
plt.scatter(clusterCenters[:,0], clusterCenters[:,1], marker="x")
plt.xlim(0, 106)
plt.ylim(0, 161)
plt.xlabel('x')
plt.ylabel('y')
plt.title("K-Means on Random Cluster points")
plt.show()


