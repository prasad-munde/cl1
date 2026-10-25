import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1 & 2: Import and Explore Dataset
df = pd.read_csv("Telco_Customer_Churn.csv")
print("Dataset imported successfully!")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
print(df.info())
print(df.describe())

# Task 3 & 4: Handle Missing Values & Duplicates
df = df.replace(r'^\s*$', np.nan, regex=True)
print("Missing values:\n", df.isnull().sum())
print("Number of duplicate rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# Task 5 & 6: Standardization & Type Conversion
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    df[column] = df[column].str.strip()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Task 7: Outlier Detection (Boxplots)
numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
plt.figure(figsize=(12, 4))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# Task 8: Feature Engineering
df["AverageMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

def count_services(row):
    services = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", 
                "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    count = 0
    for s in services:
        if s in row and row[s] == "Yes":
            count += 1
    return count

df["TotalServices"] = df.apply(count_services, axis=1)

# Task 9: Normalization / Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric_features = ["tenure", "SeniorCitizen", "MonthlyCharges", "TotalCharges", "AverageMonthlySpend", "TotalServices"]
scaled_data = scaler.fit_transform(df[numeric_features])

# Task 10: Train-Test Split
from sklearn.model_selection import train_test_split
X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Task 11: Export Cleaned Dataset
df.to_csv("cleaned_telecom_customer_churn.csv", index=False)
print("Cleaned dataset exported successfully!")