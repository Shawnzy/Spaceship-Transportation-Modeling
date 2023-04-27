import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import joblib

# Load data
df = pd.read_parquet("../../data/processed/data_new_features.parquet")

# Split into train and test sets
target = "Transported"
df_train = df.drop(columns=[target])
transported = df.Transported

X_train, X_test, y_train, y_test = train_test_split(
    df_train, transported, test_size=0.2, random_state=0
)

# Perform GridSearchCV to find best hyperparameters
rf = RandomForestClassifier(random_state=42)
param_grid = {"max_depth": range(5, 25)}
grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Train model with best hyperparameters
max_depth = grid_search.best_params_["max_depth"]
model = RandomForestClassifier(max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Export results
print(f"Model precision: {precision_score(y_test, y_pred)}")
print(f"Model recall: {recall_score(y_test, y_pred)}")

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()

# Export model
ref_cols = list(df_train.columns)
joblib.dump(value=[model, ref_cols, target], filename="../../models/model.pkl")
