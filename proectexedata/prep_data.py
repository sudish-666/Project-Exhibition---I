import pandas as pd

# cities we want in the app - add more names here later, they must match the City column exactly
CITIES = ["Delhi", "Bengaluru"]

df = pd.read_csv("city_day.csv")

for city in CITIES:
    city_df = df[df["City"] == city]
    city_df = city_df[["Date", "AQI"]].rename(columns={"Date": "date"})
    city_df = city_df.dropna()

    # one file per city, e.g. aqi_delhi.csv, aqi_bengaluru.csv
    filename = f"aqi_{city.lower()}.csv"
    city_df.to_csv(filename, index=False)
    print(f"Saved {filename} with {len(city_df)} rows for {city}")