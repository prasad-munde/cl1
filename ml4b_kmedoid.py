# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances

# Step 2: Load dataset
df = pd.read_csv("CC GENERAL.csv")
print("First 5 rows:")
print(df.head())
print("\nDataset shape:")
print(df.shape)

# Step 3: Remove customer ID
df = df.drop("CUST_ID", axis=1)

# Step 4: Handle missing values
df = df.fillna(df.median(numeric_only=True))

# Step 5: Select a sample
sample_df = df.sample(
    n=1000,
    random_state=42
).reset_index(drop=True)

# Step 6: Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(sample_df)

# K-MEDOIDS FUNCTION
def k_medoids(X, k, max_iter=100):
    # Calculate distance between all data points
    distance_matrix = pairwise_distances(X)
    # Randomly select initial medoids
    rng = np.random.default_rng(42)
    medoids = rng.choice(
        len(X),
        size=k,
        replace=False
    )
    for iteration in range(max_iter):
        # Assign every point to nearest medoid
        distances_to_medoids = distance_matrix[:, medoids]
        labels = np.argmin(
            distances_to_medoids,
            axis=1
        )
        new_medoids = medoids.copy()
        # Find new medoid for each cluster
        for cluster in range(k):
            cluster_points = np.where(
                labels == cluster
            )[0]
            if len(cluster_points) == 0:
                continue
            # Distance between points in this cluster
            cluster_distances = distance_matrix[
                np.ix_(
                    cluster_points,
                    cluster_points
                )
            ]
            # Total distance for each point
            total_distance = cluster_distances.sum(
                axis=1
            )
            # Point with minimum total distance
            new_medoids[cluster] = cluster_points[
                np.argmin(total_distance)
            ]
        # Check if medoids changed
        if np.array_equal(
            medoids,
            new_medoids
        ):
            break
        medoids = new_medoids
    # Final assignment
    distances_to_medoids = distance_matrix[:, medoids]
    labels = np.argmin(
        distances_to_medoids,
        axis=1
    )
    return labels, medoids

# SILHOUETTE METHOD
silhouette_scores = []
print("\nCalculating Silhouette Scores...")
for k in range(2, 8):
    print("K =", k)
    labels, medoids = k_medoids(
        X,
        k
    )
    score = silhouette_score(
        X,
        labels
    )
    silhouette_scores.append(score)
    print(
        "Silhouette Score:",
        round(score, 4)
    )

# FIND BEST NUMBER OF CLUSTERS
best_k = range(2, 8)[
    np.argmax(silhouette_scores)
]
print("\nBest number of clusters:", best_k)

# PLOT SILHOUETTE GRAPH
plt.figure(figsize=(8, 5))
plt.plot(
    range(2, 8),
    silhouette_scores,
    marker="o"
)
plt.title(
    "Silhouette Method for K-Medoids"
)
plt.xlabel(
    "Number of Clusters (K)"
)
plt.ylabel(
    "Silhouette Score"
)
plt.xticks(range(2, 8))
plt.grid()
plt.show()

# FINAL K-MEDOIDS
labels, medoids = k_medoids(
    X,
    best_k
)

# Add cluster labels
sample_df["Cluster"] = labels

# DISPLAY RESULTS
print("\nClustered Data:")
print(sample_df.head())
print("\nNumber of customers in each cluster:")
print(
    sample_df["Cluster"]
    .value_counts()
    .sort_index()
)

# Final silhouette score
final_score = silhouette_score(
    X,
    labels
)
print(
    "\nFinal Silhouette Score:",
    round(final_score, 4)
)

# Display medoid indices
print("\nMedoid indices:")
print(medoids)