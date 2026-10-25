# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
 accuracy_score,
 precision_score,
 recall_score,
 f1_score,
 confusion_matrix
)
from xgboost import XGBClassifier
# Step 2: Load Iris dataset
df = pd.read_csv("Iris.csv")
print("First 5 rows:")
print(df.head())
print("\nDataset shape:")
print(df.shape)
print("\nColumn names:")
print(df.columns)
# Step 3: Prepare data
# Remove Id column
df = df.drop("Id", axis=1)
# Input features
X = df[
 [
 "SepalLengthCm",
 "SepalWidthCm",
 "PetalLengthCm",
 "PetalWidthCm"
 ]
]
# Convert Species into numbers
# Iris-setosa = 0
# Iris-versicolor = 1
# Iris-virginica = 2
y = df["Species"].map({
 "Iris-setosa": 0,
 "Iris-versicolor": 1,
 "Iris-virginica": 2
})
# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
 X,
 y,
 test_size=0.20,
 random_state=42,
 stratify=y
)
print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)
# Step 5: Create models
# AdaBoost
ada_model = AdaBoostClassifier(
 estimator=DecisionTreeClassifier(max_depth=1),
 n_estimators=50,
 random_state=42
)
# Gradient Tree Boosting
gbm_model = GradientBoostingClassifier(
 n_estimators=100,
 learning_rate=0.1,
 max_depth=3,
 random_state=42
)
# XGBoost
xgb_model = XGBClassifier(
 n_estimators=100,
 max_depth=3,
 learning_rate=0.1,
 objective="multi:softmax",
 num_class=3,
 eval_metric="mlogloss",
 random_state=42
)
# Step 6: Train models
ada_model.fit(X_train, y_train)
gbm_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
# Step 7: Make predictions
ada_pred = ada_model.predict(X_test)
gbm_pred = gbm_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)
# Step 8: Calculate evaluation metrics
models = {
 "AdaBoost": ada_pred,
 "Gradient Boosting": gbm_pred,
 "XGBoost": xgb_pred
}
results = []
for model_name, predictions in models.items():
 accuracy = accuracy_score(
 y_test,
 predictions
 )
 precision = precision_score(
 y_test,
 predictions,
 average="weighted"
 )
 recall = recall_score(
 y_test,
 predictions,
 average="weighted"
 )
 f1 = f1_score(
 y_test,
 predictions,
 average="weighted"
 )
 results.append([
 model_name,
 accuracy,
 precision,
 recall,
 f1
 ])
# Step 9: Create comparison table
results_df = pd.DataFrame(
 results,
 columns=[
 "Model",
 "Accuracy",
 "Precision",
 "Recall",
 "F1 Score"
 ]
)
print("MODEL PERFORMANCE COMPARISON")
print(
 results_df.round(4)
)
# Step 10: Confusion Matrix - AdaBoost
cm_ada = confusion_matrix(
 y_test,
 ada_pred
)
print("\nAdaBoost Confusion Matrix:")
print(cm_ada)
# Step 11: Confusion Matrix - Gradient Boosting
cm_gbm = confusion_matrix(
 y_test,
 gbm_pred
)
print("\nGradient Boosting Confusion Matrix:")
print(cm_gbm)
# Step 12: Confusion Matrix - XGBoost
cm_xgb = confusion_matrix(
 y_test,
 xgb_pred
)
print("\nXGBoost Confusion Matrix:")
print(cm_xgb)
# Step 13: Plot model comparison
results_df.set_index("Model").plot(
 kind="bar",
 figsize=(10, 6)
)
plt.title(
 "Performance Comparison of Ensemble Models"
)
plt.xlabel("Models")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.legend(
 loc="lower right"
)
plt.tight_layout()
plt.show()