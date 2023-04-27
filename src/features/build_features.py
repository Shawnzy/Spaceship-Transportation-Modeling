import pandas as pd
import numpy as np


# Load data
df = pd.read_parquet("../../data/interim/cleaned_raw_data.parquet")


# Split Cabin column into 3 separate columns: CabinDeck, CabinNumber, and CabinSide
df[["CabinDeck", "CabinNumber", "CabinSide"]] = df["Cabin"].apply(
    lambda x: pd.Series(str(x).split("/"))
)


# Drop Cabin and Change CabinNumber to float
df = df.drop(columns=["Cabin"])
df.CabinNumber = df.CabinNumber.astype(float)


# Encode categorical features
features_to_encode = [
    "HomePlanet",
    "Destination",
    "CabinDeck",
    "CabinSide",
    "CryoSleep",
    "VIP",
]

df_encoded = pd.get_dummies(df[features_to_encode].astype(str))


# Drop original categorical features
df_no_categorical = df.drop(columns=features_to_encode)


# Add encoded categorical features to df_no_categorical
df_final = pd.concat([df_no_categorical, df_encoded], axis=1)


# Save final data
df_final.to_parquet("../../data/processed/data_new_features.parquet")
