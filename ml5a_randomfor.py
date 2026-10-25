# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Step 2: Load the dataset
df = pd.read_csv("car_evaluation.csv", header=None)

# Add column names
df.columns = [
    "buying",
    "maint",
    "doors",
    "persons",
    "lug_boot",
    "safety",
    "class"
]

# Step 3: Display dataset
print("First 5 rows:")
print(df.head())
print("\nDataset shape:")
print(df.shape)
print("\nColumn names:")
print(df.columns)

# Step 4: Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Step 5: Separate input and target
X = df.drop("class", axis=1)
y = df["class"]

# Step 6: Identify categorical columns
categorical_columns = X.columns

# Step 7: One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)

# Step 8: Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Step 9: Create pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ]
)

# Step 10: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# Step 11: Train the Random Forest model
pipeline.fit(X_train, y_train)

# Step 12: Make predictions
y_pred = pipeline.predict(X_test)

# Step 13: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:")
print(accuracy)
print("\nAccuracy Percentage:")
print(round(accuracy * 100, 2), "%")

# Step 14: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Step 15: Classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

# Step 16: Plot Confusion Matrix
plt.figure(figsize=(7, 5))
plt.imshow(cm)
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
classes = pipeline.classes_
plt.xticks(
    range(len(classes)),
    classes
)
plt.yticks(
    range(len(classes)),
    classes
)

# Display values
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )
plt.colorbar()
plt.show()