
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)

# Load dataset
df = pd.read_csv("dataset/thyroid.csv")
df = df.drop(
    columns=["patient_id", "dataset_name"],
    errors="ignore"
)
df = df.dropna(subset=["class"])

# Use the same 13 classes as previous experiments
counts = df["class"].value_counts()
valid_classes = counts[counts >= 10].index
df = df[df["class"].isin(valid_classes)].copy()

X = df.drop(columns=["class"])
y = df["class"]

numeric_columns = X.select_dtypes(include=["number"]).columns
categorical_columns = X.select_dtypes(exclude=["number"]).columns

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_columns),

    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_columns)
])

# Reuse the same fixed test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Base class weights: inverse class frequency
class_counts = y_train.value_counts()
n_samples = len(y_train)
n_classes = len(class_counts)

base_weights = {
    label: n_samples / (n_classes * count)
    for label, count in class_counts.items()
}

results = []
trained_models = {}

# Alpha controls the strength of class weighting.
# Alpha 0 = no weighting; alpha 1 = full inverse-frequency weighting.
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    if alpha == 0.0:
        weights = None
    else:
        weights = {
            label: weight ** alpha
            for label, weight in base_weights.items()
        }

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight=weights,
            C=1.0,
            max_iter=3000,
            random_state=42
        ))
    ])

    print(f"Training with class-weight strength alpha={alpha}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Alpha": alpha,
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "Macro F1": f1_score(
            y_test, predictions,
            average="macro",
            zero_division=0
        )
    })

    trained_models[alpha] = model

results_df = pd.DataFrame(results)
print("\n--- Weight Tuning Results ---")
print(results_df.round(4).to_string(index=False))

# Select by macro F1 for this experiment
best_alpha = results_df.loc[
    results_df["Macro F1"].idxmax(), "Alpha"
]
best_model = trained_models[best_alpha]
predictions = best_model.predict(X_test)

print(f"\nBest alpha by macro F1: {best_alpha}")
print("\nClassification report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

os.makedirs("results", exist_ok=True)
results_df.to_csv(
    "results/weight_tuning.csv",
    index=False
)
print("\nSaved results/weight_tuning.csv")
