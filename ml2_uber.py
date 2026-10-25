# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("uber.csv")
print("First 5 Rows:")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())
# Remove unnecessary columns
if 'Unnamed: 0' in df.columns:
 df.drop('Unnamed: 0', axis=1, inplace=True)
if 'key' in df.columns:
 df.drop('key', axis=1, inplace=True)
# Convert datetime
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
# Extract Date Features
df['year'] = df['pickup_datetime'].dt.year
df['month'] = df['pickup_datetime'].dt.month
df['day'] = df['pickup_datetime'].dt.day
df['hour'] = df['pickup_datetime'].dt.hour
# Drop original datetime column
df.drop('pickup_datetime', axis=1, inplace=True)
# Remove missing values
df.dropna(inplace=True)
plt.figure(figsize=(8,5))
sns.boxplot(x=df["fare_amount"])
plt.title("Boxplot of Fare Amount")
plt.show()
# Remove Outliers using IQR
Q1 = df["fare_amount"].quantile(0.25)
Q3 = df["fare_amount"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df["fare_amount"] >= lower) &
 (df["fare_amount"] <= upper)]
print("\nDataset Shape After Removing Outliers:", df.shape)
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
X = df.drop("fare_amount", axis=1)
y = df["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
 X,
 y,
 test_size=0.2,
 random_state=42
)
linear = LinearRegression()
linear.fit(X_train, y_train)
pred_linear = linear.predict(X_test)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
pred_ridge = ridge.predict(X_test)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
pred_lasso = lasso.predict(X_test)
def evaluate_model(name, actual, predicted):
 rmse = np.sqrt(mean_squared_error(actual, predicted))
 mae = mean_absolute_error(actual, predicted)
 r2 = r2_score(actual, predicted)
 print("\n==============================")
 print(name)
 print("==============================")
 print("R2 Score :", round(r2,4))
 print("RMSE :", round(rmse,4))
 print("MAE :", round(mae,4))
evaluate_model("Linear Regression", y_test, pred_linear)
evaluate_model("Ridge Regression", y_test, pred_ridge)
evaluate_model("Lasso Regression", y_test, pred_lasso)