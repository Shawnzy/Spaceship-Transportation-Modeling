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

# Load data

# Validate Data

# Clean Data

# Build Features

# Validate Final Data

# Create X_features

# Import Model and Predict

# Print Scores

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Model precision: {precision_score(y_test, y_pred)}")
print(f"Model recall: {recall_score(y_test, y_pred)}")

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
