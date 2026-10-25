import pandas as pd
import numpy as np
from scipy.stats import mode, skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)

# Define column names since the CSV file has no header row
column_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

# Load dataset with header=None and assign column names
data = pd.read_csv("diabetes.csv", header=None, names=column_names)

# UNIVARIATE ANALYSIS
print("\nUNIVARIATE ANALYSIS")
summary = pd.DataFrame()
summary["Mean"] = data.mean(numeric_only=True)
summary["Median"] = data.median(numeric_only=True)
summary["Mode"] = data.mode().iloc[0]
summary["Variance"] = data.var(numeric_only=True)
summary["Standard Deviation"] = data.std(numeric_only=True)
summary["Skewness"] = data.skew(numeric_only=True)
summary["Kurtosis"] = data.kurt(numeric_only=True)
print(summary)

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# LINEAR REGRESSION
print("\nLINEAR REGRESSION")
linear = LinearRegression()
linear.fit(X_train, y_train)
pred_linear = linear.predict(X_test)
print("R2 Score :", r2_score(y_test, pred_linear))

# LOGISTIC REGRESSION
print("\nLOGISTIC REGRESSION")
logistic = LogisticRegression(max_iter=1000)
logistic.fit(X_train, y_train)
pred_log = logistic.predict(X_test)
print("Accuracy :", accuracy_score(y_test, pred_log))

# C. MULTIPLE REGRESSION
print("\nMULTIPLE REGRESSION")
multiple = LinearRegression()
multiple.fit(X_train, y_train)
pred_multiple = multiple.predict(X_test)
print("Multiple Regression R2 Score:", r2_score(y_test, pred_multiple))

# D. Comparison
print("\nCOMPARISON")
print("Linear Regression R2 Score :", r2_score(y_test, pred_linear))
print("Logistic Regression Accuracy :", accuracy_score(y_test, pred_log))