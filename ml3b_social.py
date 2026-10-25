# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
 confusion_matrix,
 accuracy_score,
 precision_score,
 recall_score,
 classification_report
)
# Step 2: Load the dataset
df = pd.read_csv("Social_Network_Ads.csv")
# Step 3: Display the dataset
print("First 5 rows:")
print(df.head())
print("\nDataset shape:")
print(df.shape)
# Step 4: Check for missing values
print("\nMissing values:")
print(df.isnull().sum())
# Step 5: Select input and output variables
# Age and EstimatedSalary are the input features
X = df[["Age", "EstimatedSalary"]]
# Purchased is the target variable
y = df["Purchased"]
# Step 6: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
 X,
 y,
 test_size=0.25,
 random_state=42
)
print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)
# Step 7: Feature Scaling
# Important for K-NN because it uses distance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Step 8: Create K-NN model
knn = KNeighborsClassifier(n_neighbors=5)
# Step 9: Train the model
knn.fit(X_train, y_train)
# Step 10: Make predictions
y_pred = knn.predict(X_test)
# Step 11: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
# Step 12: Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
# Step 13: Error Rate
error_rate = 1 - accuracy
print("Error Rate:", error_rate)
# Step 14: Precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)
# Step 15: Recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)
# Step 16: Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Step 17: Display Confusion Matrix as a graph
plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["Not Purchased", "Purchased"])
plt.yticks([0, 1], ["Not Purchased", "Purchased"])
# Display values inside matrix
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j],
            ha="center",
            va="center")
plt.colorbar()
plt.show()
