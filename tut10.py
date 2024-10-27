from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances


'''Task 1'''
iris = pd.read_csv('iris.csv')
features = iris.drop('variety',axis=1)
labels = iris['variety']

def transformLabels(y):
    if y == "Setosa":
        return 1
    elif y == "Versicolor":
        return 0
    return 2

labels = labels.apply(transformLabels)

hier_cluster = AgglomerativeClustering(n_clusters=3, linkage='average')
hier_cluster.fit(features)
predictedLabels = hier_cluster.labels_

print(hier_cluster.labels_)

'''Exercise 1'''
plt.figure(dpi=150)
plt.scatter(x=iris["petal.width"], y=iris["petal.length"], 
            c=labels, marker='o', label="Original Labels")
plt.xlabel('petal width')
plt.ylabel('petal length')
plt.legend()
plt.show()

plt.figure(dpi=150)
plt.scatter(x=iris["petal.width"], y=iris["petal.length"], 
            c=predictedLabels, marker='^', label="PredictedValues")
plt.xlabel('petal width')
plt.ylabel('petal length')
plt.legend()
plt.show()

"""
Plotting side by side
"""
# plt.figure(dpi=150)
# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.scatter(x=iris["petal.width"], y=iris["petal.length"], 
#             c=labels, marker='o', label="Original Labels")
# ax1.set_xlabel('petal width')
# ax1.set_ylabel('petal length')
# ax1.legend()

# ax2.scatter(x=iris["petal.width"], y=iris["petal.length"], 
#             c=predictedLabels, marker='^', label="PredictedValues")
# ax2.set_xlabel('petal width')
# ax2.set_ylabel('petal length')
# ax2.legend()
# plt.show()


"""
Task 2: Part A
"""
#Dataset 1 (the last dataset in Tutorial 8)
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

sample_pts = np.concatenate((rect_pts1, rect_pts2, 
                              rect_pts3, rect_pts4), axis=0)


hier_cluster = AgglomerativeClustering(n_clusters=4, linkage='single')

dist_matrix = euclidean_distances(sample_pts)
# hier_cluster.fit(dist_matrix)
hier_cluster.fit(sample_pts)

plt.figure(dpi=150)
plt.scatter(sample_pts[:, 0], sample_pts[:, 1], 
            c=hier_cluster.labels_, 
            cmap=plt.cm.Set1)
plt.xlim(0, 106)
plt.ylim(0, 161)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

"""
Task 2: Part B
"""
# Generate Dataset 2
rad = np.random.rand(400)
ang = 2 * np.pi * np.random.rand(400)
pt_x = np.multiply(rad, np.cos(ang))
pt_y = np.multiply(rad, np.sin(ang))
disc_pts1 = np.c_[pt_x, pt_y]

rad = np.random.rand(400)
ang = 2 * np.pi * np.random.rand(400)
pt_x = np.multiply(rad, np.cos(ang)) + 1.8
pt_y = np.multiply(rad, np.sin(ang))
disc_pts2 = np.c_[pt_x, pt_y]

disc_pts = np.append(disc_pts1, disc_pts2, axis=0)

#Visualization of dataset 2
plt.figure(dpi=150)
plt.scatter(disc_pts[:, 0], disc_pts[:, 1])
plt.xlim(-2, 6)
plt.ylim(-2, 2)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#Calculate pairwise Euclidean distances
dist_mtrx = euclidean_distances(disc_pts)

#Single linkage
hier_cluster = AgglomerativeClustering(n_clusters=2, linkage='single')
hier_cluster.fit(dist_mtrx)

plt.figure(dpi=150)
plt.scatter(disc_pts[:, 0], disc_pts[:, 1], 
            c=hier_cluster.labels_, cmap=plt.cm.Set1)
plt.xlim(-2, 6)
plt.ylim(-2, 2)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


#Complete linkage (better than single linkage but worse than k-means)
hier_cluster = AgglomerativeClustering(n_clusters=2, linkage='complete')
hier_cluster.fit(dist_mtrx)

plt.figure(dpi=150)
plt.scatter(disc_pts[:, 0], disc_pts[:, 1], 
            c=hier_cluster.labels_, cmap=plt.cm.Set1)
plt.xlim(-2, 6)
plt.ylim(-2, 2)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Heirarchical Clustering")
plt.show()

#k-means (the best results)
kmeans_estimator = KMeans(n_clusters=2, random_state=0)
k_clusters = kmeans_estimator.fit(disc_pts)

plt.figure(dpi=150)
plt.scatter(disc_pts[:, 0], disc_pts[:, 1], 
            c=k_clusters.labels_, cmap=plt.cm.Set1)
plt.xlim(-2, 6)
plt.ylim(-2, 2)
plt.xlabel('x')
plt.ylabel('y')
plt.title("K-means Clustering")
plt.show()


"""
Hierarchical clustering with multiple linkages
"""

linkages = ["ward", "complete", "average", "single"]

def partB(linkage):
    hier_cluster = AgglomerativeClustering(n_clusters=2, linkage=linkage)
    hier_cluster.fit(dist_mtrx)
    
    plt.figure(dpi=150)
    plt.scatter(disc_pts[:, 0], disc_pts[:, 1], 
                c=hier_cluster.labels_, cmap=plt.cm.Set1)
    plt.xlim(-2, 6)
    plt.ylim(-2, 2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"Hierarchical clustering with {linkage} linkage")
    plt.show()

"""
Exercise 3
"""
for linkage in linkages:
    partB(linkage)