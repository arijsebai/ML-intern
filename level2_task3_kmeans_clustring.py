# step 1 import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# step 2 load the data
df = pd.read_csv("Data Set For Task/1) iris.csv")
print(df.head())
print(df.info())
print("dataset shape:", df.shape)
print("Missing values:", df.isnull().sum())

#step 3 slecting relevant features for clustering
features= ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = df[features]
print("features for clustering:", features)

#step 4 Scale the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

#step 5 Determine the optimal number of clusters using the elbow method
inertia_values = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(x_scaled)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("level2_task3_kmeans_clustring_elbow_method.png", dpi=300)
plt.show()

#step 6 train the final k-means model
best_k=3
kmeans_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster"]= kmeans_model.fit_predict(x_scaled)
print ("cluster counts:\n", df["cluster"].value_counts().sort_index())

#step 7 reduce dimensionality for visualization using PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)
df["pca1"] = x_pca[:, 0]
df["pca2"] = x_pca[:, 1]

#step 8 visualize the clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="pca1", y="pca2", hue="cluster", palette="viridis", s=80)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clustering of Iris Dataset (PCA Visualization)")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("level2_task3_kmeans_clustring_pca_visualization.png", dpi=300)
plt.show()

#step 9 interpret the clusters
cluster_interpretation = pd.crosstab(df["cluster"], df["species"])
print("Cluster Interpretation:\n", cluster_interpretation)
print("clustercenters in original features scale:")
cluster_centers = scaler.inverse_transform(kmeans_model.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)
print(cluster_centers_df)
