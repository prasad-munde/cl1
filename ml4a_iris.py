# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Step 2: Load the dataset
df = pd.read_csv("Iris.csv")
# Step 3: Check dataset information
print("\nDataset shape:")
print(df.shape)
print("\nColumn names:")
print(df.columns)
# Step 4: Select features for clustering
# We don't use Id or Species for clustering
X = df[[
 "SepalLengthCm",
 "SepalWidthCm",
 "PetalLengthCm",
 "PetalWidthCm"
]]
print("\nFeatures used for clustering:")
print(X.head())
# Step 5: Elbow Method
wcss = []
for i in range(1, 11):
 kmeans = KMeans(
 n_clusters=i,
 random_state=42,
 n_init=10
 )

 kmeans.fit(X)
 wcss.append(kmeans.inertia_)
# Step 6: Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(
 range(1, 11),
 wcss,
 marker="o"
)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.xticks(range(1, 11))
plt.grid()
plt.show()
# Step 7: Create K-Means model
# From the elbow graph, choose 3 clusters
kmeans = KMeans(
 n_clusters=3,
 random_state=42,
 n_init=10
)
# Step 8: Fit the model
kmeans.fit(X)
# Step 9: Get cluster labels
clusters = kmeans.labels_
# Add cluster information to dataset
df["Cluster"] = clusters
# Step 10: Display cluster results
print("\nCluster results:")
print(df[[
 "SepalLengthCm",
 "SepalWidthCm",
 "PetalLengthCm",
 "PetalWidthCm",
 "Cluster"
]].head(10))
# Step 11: Display number of points in each cluster
print("\nNumber of data points in each cluster:")
print(df["Cluster"].value_counts())
# Step 12: Visualize the clusters
plt.figure(figsize=(8, 5))
plt.scatter(
 X["PetalLengthCm"],
 X["PetalWidthCm"],
 c=clusters
)
# Plot cluster centers
centers = kmeans.cluster_centers_
plt.scatter(
 centers[:, 2],
 centers[:, 3],
 marker="X",
 s=200,
 label="Centroids"
)
plt.title("K-Means Clustering on Iris Dataset")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()
plt.grid()
plt.show()