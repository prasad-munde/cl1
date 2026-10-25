# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load Dataset
df = pd.read_csv("Iris.csv")
# Display first five rows
print(df.head())
# Dataset Shape
print("\nDataset Shape:", df.shape)
# Remove Id column
df = df.drop("Id", axis=1)
# Features and Target
X = df.drop("Species", axis=1)
y = df["Species"]
# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42)
# Create LDA Model
lda = LinearDiscriminantAnalysis()
# Train Model
lda.fit(X_train, y_train)
# Predict
y_pred = lda.predict(X_test)
# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))
# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))