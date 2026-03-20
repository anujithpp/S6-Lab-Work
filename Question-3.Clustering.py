# ============================================
# IMPORTS
# ============================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ============================================
# LOAD DATASET (YOUR FILE)
# ============================================
df = pd.read_csv("/content/supermarket_sales - Sheet1.csv")

print(df.head())

# ============================================
# (a) APPLY K-MEANS CLUSTERING
# ============================================

# Select useful features
data = df[['Total', 'Quantity', 'Rating', 'Gender', 'Customer type']]

# Convert categorical → numeric
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Customer type'] = le.fit_transform(data['Customer type'])

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(data)

# ============================================
# (b) ELBOW METHOD
# ============================================

wcss = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,10), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Choose optimal k (usually 3 or 4 from graph)
k = 3

# ============================================
# APPLY FINAL K-MEANS
# ============================================

kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

print("\nCluster Counts:")
print(df['Cluster'].value_counts())

# ============================================
# (c) ANALYZE CLUSTERS
# ============================================

print("\nCluster Characteristics:")
print(df.groupby('Cluster')[['Total', 'Quantity', 'Rating']].mean())

# ============================================
# (d) VISUALIZATION (PCA)
# ============================================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:,0], X_pca[:,1], c=df['Cluster'])
plt.title("Customer Clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
