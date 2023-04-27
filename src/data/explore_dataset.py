import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


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

# Split Cabin column into 3 separate columns: CabinDeck, CabinNumber, and CabinSide
df[["CabinDeck", "CabinNumber", "CabinSide"]] = df["Cabin"].apply(
    lambda x: pd.Series(str(x).split("/"))
)

df.CabinNumber = df.CabinNumber.astype(float)
df.Transported = df.Transported.astype(bool)

# Drop the following columns: PassengerId, Name, and Cabin
df = df.drop(columns=["PassengerId", "Name", "Cabin"])

# Drop all rows with NaN values
df = df.dropna()

features_to_encode = [
    "HomePlanet",
    "Destination",
    "CabinDeck",
    "CabinSide",
    "CryoSleep",
    "VIP",
]
df[features_to_encode].info()

df_encoded = pd.get_dummies(df[features_to_encode].astype(str))

df_no_categorical = df.drop(columns=features_to_encode)

df_final = pd.concat([df_no_categorical, df_encoded], axis=1)


df_train = df_final.drop(columns=["Transported"])
transported = df_final.Transported

X_train, X_test, y_train, y_test = train_test_split(
    df_train, transported, test_size=0.2, random_state=0
)

rf = RandomForestClassifier(random_state=42)
param_grid = {"max_depth": range(5, 25)}
grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X_train, y_train)
grid_search.best_params_

model = RandomForestClassifier(max_depth=20, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Model precision: {precision_score(y_test, y_pred)}")
print(f"Model recall: {recall_score(y_test, y_pred)}")

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
