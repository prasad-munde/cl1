import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1 & 2: API Request for Current Weather
API_KEY = "your_api_key_here"
cities = ["Pune", "Mumbai", "Delhi", "Bengaluru", "Chennai"]

weather_list = []
for city in cities:
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather_list.append({
            "City": city,
            "Temperature": data["main"]["temp"],
            "Feels Like": data["main"]["feels_like"],
            "Humidity": data["main"]["humidity"],
            "Pressure": data["main"]["pressure"],
            "Wind Speed": data["wind"]["speed"],
            "Rainfall": data.get("rain", {}).get("1h", 0),
            "Weather": data["weather"][0]["description"],
            "Latitude": data["coord"]["lat"],
            "Longitude": data["coord"]["lon"]
        })
    else:
        print(f"Could not get data for {city}")

print("Weather data collected successfully!")
df = pd.DataFrame(weather_list)

# Task 3: Explore Structure
print("Shape of data:", df.shape)
print("Column Names:", df.columns.tolist())
print("Data Types:\n", df.dtypes)
print(df.head())

# Task 4: Cleaning & Preprocessing
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicates removed.")

# Task 5: Data Modeling & Statistics
print("Highest Temperature City:", df.loc[df["Temperature"].idxmax()]["City"], df["Temperature"].max())
print("Lowest Temperature City:", df.loc[df["Temperature"].idxmin()]["City"], df["Temperature"].min())
print("Highest Humidity City:", df.loc[df["Humidity"].idxmax()]["City"], df["Humidity"].max())
print("Lowest Humidity City:", df.loc[df["Humidity"].idxmin()]["City"], df["Humidity"].min())

print("Average Temperature:", df["Temperature"].mean())
print("Maximum Temperature:", df["Temperature"].max())
print("Minimum Temperature:", df["Temperature"].min())
print("Average Humidity:", df["Humidity"].mean())
print("Average Wind Speed:", df["Wind Speed"].mean())

# Task 6: Visualizations (Bar Plots)
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="City", y="Temperature")
plt.title("Temperature Comparison of Cities")
plt.xlabel("City")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="City", y="Humidity")
plt.title("Humidity Comparison of Cities")
plt.xlabel("City")
plt.ylabel("Humidity (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="City", y="Wind Speed")
plt.title("Wind Speed Comparison of Cities")
plt.xlabel("City")
plt.ylabel("Wind Speed (m/s)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Task 7: Forecast Data Retrieval & Aggregation
forecast_url = "http://api.openweathermap.org/data/2.5/forecast"
forecast_list = []
for city in cities:
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    response = requests.get(forecast_url, params=params)
    if response.status_code == 200:
        data = response.json()
        for item in data["list"]:
            forecast_list.append({
                "City": city,
                "Date": item["dt_txt"],
                "Temperature": item["main"]["temp"],
                "Humidity": item["main"]["humidity"],
                "Pressure": item["main"]["pressure"],
                "Wind Speed": item["wind"]["speed"],
                "Rainfall": item.get("rain", {}).get("3h", 0),
                "Weather": item["weather"][0]["description"]
            })

forecast_df = pd.DataFrame(forecast_list)
forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
forecast_df = forecast_df.drop_duplicates()

# Temperature Trend Line Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=forecast_df, x="Date", y="Temperature", hue="City")
plt.title("Temperature Trend of Different Cities")
plt.xlabel("Date and Time")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Task 8: Geospatial Map with Folium
import folium
weather_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="OpenStreetMap")
for i in range(len(df)):
    city = df.loc[i, "City"]
    lat = df.loc[i, "Latitude"]
    lon = df.loc[i, "Longitude"]
    temp = df.loc[i, "Temperature"]
    folium.Marker(location=[lat, lon], popup=f"{city}: {temp}°C", tooltip=city).add_to(weather_map)
weather_map.save("weather_map.html")
print("Map created successfully!")

# Task 9: Correlation and Heatmap
correlation = forecast_df[["Temperature", "Humidity", "Pressure", "Wind Speed", "Rainfall"]].corr()
print(correlation)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Weather Attributes")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=forecast_df, x="Temperature", y="Humidity", hue="City")
plt.title("Temperature vs Humidity")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.tight_layout()
plt.show()