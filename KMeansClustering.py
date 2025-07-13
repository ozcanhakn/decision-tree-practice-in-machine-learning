from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X , _ = make_blobs(n_samples =300, centers = 4,cluster_std=0.6,random_state=42)

plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.title("Örnek Veri")

k_means = KMeans(n_clusters=4)
k_means.fit(X)

labels = k_means.labels_

plt.figure()
plt.scatter(X[:,0], X[:,1], c=labels, cmap ="viridis")

centers = k_means.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c="red", marker="X")
plt.title("K-means")