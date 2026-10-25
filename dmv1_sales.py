import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Data from CSV and JSON file formats
csv_data = pd.read_csv('sales.csv')
json_data = pd.read_json('sales.json')

print("CSV Data")
print(csv_data.head())
print("JSON Data")
print(json_data.head())

# Combine/Concatenate available datasets into a unified dataframe
sales = pd.concat([csv_data, json_data], ignore_index=True)
print(sales.head())

# 2. Explore Data Structure and Quality
print(sales.tail())
print("Dataset Shape")
print(sales.shape)
print("Column Names")
print(sales.columns)
print("Dataset Information")
sales.info()
print("Missing Values")
print(sales.isnull().sum())
print("Duplicate Records")
print(sales.duplicated().sum())
print("Summary Statistics")
print(sales.describe())

# 3. Data Cleaning
print("Missing Values Before Cleaning:")
print(sales.isnull().sum())

print("Duplicate Records Before Cleaning:")
print(sales.duplicated().sum())

# Remove duplicate records
sales.drop_duplicates(inplace=True)

print("Missing Values After Removing Duplicates:")
print(sales.isnull().sum())

# Drop rows with missing values
sales.dropna(inplace=True)

print("Dataset after Cleaning:")
print(sales.head())

print("Remaining Missing Values:")
print(sales.isnull().sum())

# 4. Data Transformation
# Create new variable TotalSales
sales["TotalSales"] = sales["Quantity"] * sales["Price"]
print("TotalSales column created successfully!")
print(sales.head())

print("Columns in Dataset")
print(sales.columns)

print("Data Types")
print(sales.dtypes)

# 5. Sales Data Analysis & Metrics
print("Summary Statistics")
print(sales.describe())

total_revenue = sales["TotalSales"].sum()
print("Total Revenue:", total_revenue)

average_order_value = sales["TotalSales"].mean()
print("Average Order Value:", average_order_value)

# Sales by Category
category_sales = sales.groupby("Category")["TotalSales"].sum()
print("Sales by Category")
print(category_sales)

# Product-wise Sales
product_sales = sales.groupby("Product")["TotalSales"].sum()
print("Product-wise Sales")
print(product_sales)

# Top Selling Products
top_products = sales.groupby("Product")["TotalSales"].sum().sort_values(ascending=False)
print("Top Selling Products")
print(top_products)

# Average Price by Category
average_price = sales.groupby("Category")["Price"].mean()
print("Average Price by Category")
print(average_price)

# Product Category Distribution
category_distribution = sales["Category"].value_counts()
print("Product Category Distribution")
print(category_distribution)

# 6. Data Visualizations
# Bar Plot: Sales by Category
category_sales.plot(kind='bar')
plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Bar Plot: Top Selling Products
top_products.plot(kind='bar')
plt.title("Top Selling Products")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Pie Chart: Product Category Distribution
category_distribution.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title("Product Category Distribution")
plt.ylabel("")
plt.show()

# 7. Exporting Cleaned Data
sales.to_csv('combined_sales.csv', index=False)
print("combined_sales.csv created successfully!")