# Data Aggregation
# Analyzing Sales Performance by Region in a Retail Company
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Import the dataset[cite: 57]
folder = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(folder, "Retail_Sales_Dataset.csv"))

# 2. Explore the dataset[cite: 57]
print("First 5 rows:")
print(df.head())
print("\nDataset Information:")
df.info()
print("\nStatistical Summary:")
print(df.describe())
print("\nColumn Names:")
print(df.columns.tolist())
print("\nMissing Values:")
print(df.isnull().sum())

# Remove extra spaces from column names[cite: 57]
df.columns = df.columns.str.strip()

# 3. Identify relevant variables[cite: 57]
print("\nRelevant columns for aggregation:")
print(df.columns.tolist())
region_col = "Region"
sales_col = "Revenue"
category_col = "Category"

# 4. Group sales by region[cite: 58]
region_sales = df.groupby(region_col)[sales_col].sum()
print("\nTotal Sales by Region:")
print(region_sales)

# 5. Visualize sales distribution by region[cite: 58]
region_sales.plot(kind="bar", color="skyblue", figsize=(8, 5))
plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales Amount")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Pie chart of sales distribution[cite: 58]
region_sales.plot(kind="pie", autopct="%1.1f%%", figsize=(7, 7))
plt.title("Regional Sales Distribution")
plt.ylabel("")
plt.tight_layout()
plt.show()

# 6. Identify top-performing regions[cite: 58]
print("\nTop Performing Regions:")
print(region_sales.sort_values(ascending=False))
print("\nBest Performing Region:")
print(region_sales.idxmax())
print("Highest Sales:", region_sales.max())

# 7. Group sales by region and product category[cite: 58]
category_sales = df.groupby(
    [region_col, category_col]
)[sales_col].sum().unstack(fill_value=0)
print("\nSales by Region and Product Category:")
print(category_sales)

# 8. Stacked bar chart[cite: 58, 59]
category_sales.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)
plt.title("Sales by Region and Product Category")
plt.xlabel("Region")
plt.ylabel("Total Sales Amount")
plt.legend(title="Product Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Grouped bar chart[cite: 59]
category_sales.plot(
    kind="bar",
    stacked=False,
    figsize=(10, 6)
)
plt.title("Regional Sales Comparison by Product Category")
plt.xlabel("Region")
plt.ylabel("Total Sales Amount")
plt.legend(title="Product Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nData Aggregation Completed Successfully!")[cite: 59]