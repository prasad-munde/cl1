import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Import and Clean Column Names
df = pd.read_csv("RealEstate_Prices.csv")
print("Dataset imported successfully!")
print("Dataset shape:", df.shape)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print(df.columns.tolist())

# Task 2: Handle Missing Values
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage)

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Task 4: Filter and Subset Data
young_houses = df[df["x2_house_age"] < 10]
print("Number of young houses:", len(young_houses))

mrt_close = df[df["x3_distance_to_the_nearest_mrt_station"] < 500]
print("Properties within 500 meters of MRT:", len(mrt_close))

many_stores = df[df["x4_number_of_convenience_stores"] > 5]
print("Properties with more than 5 convenience stores:", len(many_stores))

median_price = df["y_house_price_of_unit_area"].median()
high_price = df[df["y_house_price_of_unit_area"] > median_price]
print("Median house price:", median_price)
print("Properties above median price:", len(high_price))

# Task 6: Aggregations and Summary Statistics
average_price = df["y_house_price_of_unit_area"].mean()
print("Average house price per unit area:", average_price)
print("Minimum house price:", df["y_house_price_of_unit_area"].min())
print("Maximum house price:", df["y_house_price_of_unit_area"].max())

price_by_stores = df.groupby("x4_number_of_convenience_stores")["y_house_price_of_unit_area"].agg(
    ['count', 'mean', 'min', 'max']
).reset_index()
print(price_by_stores)

plt.figure(figsize=(10, 6))
sns.barplot(data=price_by_stores, x="x4_number_of_convenience_stores", y="mean")
plt.title("Average House Price by Number of Convenience Stores")
plt.xlabel("Number of Convenience Stores")
plt.ylabel("Average House Price per Unit Area")
plt.tight_layout()
plt.show()

# Task 7: Outlier Detection
numeric_columns = ["x2_house_age", "x3_distance_to_the_nearest_mrt_station", "x4_number_of_convenience_stores", "y_house_price_of_unit_area"]
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()