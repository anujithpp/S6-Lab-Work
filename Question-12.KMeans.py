# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Install pyclustering and scikit-fuzzy if not already installed
!pip install -q pyclustering scikit-fuzzy

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# K-Medoids via pyclustering
from pyclustering.cluster.kmedoids import kmedoids

# Fuzzy C-Means
import skfuzzy as fuzz

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
n_clusters = 3

# ------------------- K-Means -------------------
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels_kmeans = kmeans.fit_predict(X)
silhouette_kmeans = silhouette_score(X, labels_kmeans)

# ------------------- K-Medoids -------------------
# Initial medoid indices
initial_medoids = [0, 50, 100]
kmedoids_instance = kmedoids(X, initial_medoids)
kmedoids_instance.process()
clusters = kmedoids_instance.get_clusters()

# Convert cluster lists to flat label array
labels_kmedoids = np.zeros(X.shape[0])
for cluster_id, cluster_points in enumerate(clusters):
    for idx in cluster_points:
        labels_kmedoids[idx] = cluster_id
silhouette_kmedoids = silhouette_score(X, labels_kmedoids)

# ------------------- Fuzzy C-Means -------------------
X_T = X.T  # skfuzzy expects features x samples
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    X_T, c=n_clusters, m=2, error=0.005, maxiter=1000
)
labels_fcm = np.argmax(u, axis=0)
silhouette_fcm = silhouette_score(X, labels_fcm)

# ------------------- Print Results -------------------
print("Silhouette Scores:")
print(f"K-Means: {silhouette_kmeans:.3f}")
print(f"K-Medoids: {silhouette_kmedoids:.3f}")
print(f"Fuzzy C-Means: {silhouette_fcm:.3f}")

# ------------------- Optional 2D Visualization -------------------
X_2D = PCA(n_components=2).fit_transform(X)

plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.scatter(X_2D[:,0], X_2D[:,1], c=labels_kmeans, cmap='viridis', s=50)
plt.title('K-Means Clustering')

plt.subplot(1,3,2)
plt.scatter(X_2D[:,0], X_2D[:,1], c=labels_kmedoids, cmap='viridis', s=50)
plt.title('K-Medoids Clustering')

plt.subplot(1,3,3)
plt.scatter(X_2D[:,0], X_2D[:,1], c=labels_fcm, cmap='viridis', s=50)
plt.title('Fuzzy C-Means Clustering')

plt.show()
