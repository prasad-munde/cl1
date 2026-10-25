# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Load Dataset
df = pd.read_csv("Wine.csv")
# Display dataset
print(df.head())
print("\nDataset Shape:", df.shape)
# Separate Features and Target
X = df.drop("Customer_Segment", axis=1)
y = df["Customer_Segment"]
# Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
# Explained Variance
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)
print("\nTotal Variance Explained:",
 pca.explained_variance_ratio_.sum())
# Create DataFrame
pca_df = pd.DataFrame(X_pca,
 columns=["Principal Component 1",
 "Principal Component 2"])
pca_df["Customer_Segment"] = y
# Plot
plt.figure(figsize=(8,6))
colors = ['red', 'green', 'blue']
for segment, color in zip(sorted(pca_df["Customer_Segment"].unique()), colors):
 subset = pca_df[pca_df["Customer_Segment"] == segment]
 plt.scatter(subset["Principal Component 1"],
 subset["Principal Component 2"],
 color=color,
 label=f"Segment {segment}")
plt.title("PCA of Wine Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()