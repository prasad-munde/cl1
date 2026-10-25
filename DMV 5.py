import os
import pandas as pd
import matplotlib.pyplot as plt
# Get the folder where p5.py is saved
folder = os.path.dirname(os.path.abspath(__file__))
# Load the dataset from the same folder
df = pd.read_csv(os.path.join(folder, "Mumbai_AQI_Dataset.csv"))
print(df.head())
print("\nDataset Information:")
df.info()
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())
# 3. Identify relevant variables
df["Date"] = pd.to_datetime(
 df["Date"], format="%d/%m/%y", errors="coerce"
)
df = df.dropna(subset=["Date", "AQI"])
df = df.sort_values("Date")
print("\nRelevant Columns:")
print(["Date", "AQI", "PM2.5", "PM10", "CO"])
# 4. Line plot: Overall AQI trend
plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["AQI"], color="blue",
 label="AQI", linewidth=1.5)
plt.title("Overall AQI Trend in Mumbai")
plt.xlabel("Date")
plt.ylabel("Air Quality Index")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# 5. Individual pollutant trends
pollutants = ["PM2.5", "PM10", "CO"]
for pollutant in pollutants:
 plt.figure(figsize=(12, 5))
 plt.plot(df["Date"], df[pollutant],
 label=pollutant, linewidth=1.5)
 plt.title(pollutant + " Levels Over Time in Mumbai")
 plt.xlabel("Date")
 plt.ylabel("Pollutant Level")
 plt.legend()
 plt.grid(True, alpha=0.3)
 plt.xticks(rotation=45)
 plt.tight_layout()
 plt.show()
# 6. Bar plot: Average AQI by month
df["Month"] = df["Date"].dt.to_period("M").astype(str)
monthly_aqi = df.groupby("Month")["AQI"].mean()
plt.figure(figsize=(12, 5))
plt.bar(monthly_aqi.index, monthly_aqi.values,
 color="skyblue", label="Average AQI")
plt.title("Monthly Average AQI in Mumbai")
plt.xlabel("Month")
plt.ylabel("Average AQI")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
# 7. Boxplots: AQI and pollutant distributions
plt.figure(figsize=(7, 5))
plt.boxplot(df["AQI"].dropna(), labels=["AQI"])
plt.title("Distribution of AQI Values")
plt.ylabel("AQI")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
plt.figure(figsize=(9, 5))
plt.boxplot(
 [df[col].dropna() for col in pollutants],
 labels=pollutants
)
plt.title("Distribution of Pollutant Levels")
plt.xlabel("Pollutants")
plt.ylabel("Measured Level")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
# 8. Scatter plots: AQI versus pollutants
for pollutant in pollutants:
 plt.figure(figsize=(7, 5))
 plt.scatter(df[pollutant], df["AQI"],
 color="green", alpha=0.5,
 label=pollutant)
 plt.title("AQI vs " + pollutant)
 plt.xlabel(pollutant + " Level")
 plt.ylabel("Air Quality Index")
 plt.legend()
 plt.grid(True, alpha=0.3)
 plt.tight_layout()
 plt.show()
# 9. Visualization completed
print("\nAll AQI visualizations completed successfully!")