import pandas as pd
import numpy as np

# This code reads the spaceship passengers csv file
df = pd.read_csv("../../data/raw/spaceship_passengers.csv")

# Explore the data
df.info()
df.describe()

# For Destination, fill in NaN values with the most common value
most_com_val = df.Destination.value_counts().idxmax()
df.Destination = df.Destination.fillna(most_com_val)

# For VIP, fill in NaN values with the most common value
most_com_val = df.VIP.value_counts().idxmax()
df.VIP = df.VIP.fillna(most_com_val)

# This code replaces NaN values in the columns 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', and 'VRDeck' with their respective median values.
cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
df[cols] = df[cols].fillna(df[cols].median())

# Change Dtype of Transported to bool
df.Transported = df.Transported.astype(bool)

# Drop the following columns: PassengerId, Name, and Cabin
df = df.drop(columns=["PassengerId", "Name"])

# Drop all rows with NaN values
df = df.dropna()

# Save cleaned data to parquet file
df.to_parquet("../../data/interim/cleaned_raw_data.parquet")
