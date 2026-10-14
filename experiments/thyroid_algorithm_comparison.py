
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)

# 1. Load the dataset
df = pd.read_csv("dataset/thyroid.csv")
df = df.drop(
    columns=["patient_id", "dataset_name"],
    errors="ignore"
)
df = df.dropna(subset=["class"])

# Use classes with at least 10 records for this comparison.
# Rare classes remain in the original dataset but are excluded
# from this particular benchmark because their test results
# would be based on very few examples.
class_counts = df["class"].value_counts()
valid_classes = class_counts[class_counts >= 10].index
df = df[df["class"].isin(valid_classes)].copy()

X = df.drop(columns=["class"])
y = df["class"]

print("Number of evaluated classes:", y.nunique())
print("Records used:", len(df))

# 2. Identify column types
numeric_columns = X.select_dtypes(include=["number"]).columns
categorical_columns = X.select_dtypes(exclude=["number"]).columns

# 3. Build preprocessing pipeline
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

# 4. Make one shared train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Define the models
models = {
    "Majority Baseline": DummyClassifier(
        strategy="most_frequent"
    ),

    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "Extra Trees": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
}

# 6. Train and evaluate
results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "Macro F1": f1_score(
            y_test, predictions,
            average="macro",
            zero_division=0
        )
    }

    results.append(metrics)
    trained_models[name] = model

# 7. Display comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Macro F1", ascending=False
)

print("\n--- Algorithm Comparison ---")
print(results_df.round(4).to_string(index=False))

# 8. Print detailed report for the highest macro F1
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]
best_predictions = best_model.predict(X_test)

print("\nBest model by macro F1:", best_name)
print("\nClassification report:")
print(classification_report(
    y_test,
    best_predictions,
    zero_division=0
))

# 9. Save the comparison
os.makedirs("results", exist_ok=True)
results_df.to_csv(
    "results/algorithm_comparison.csv",
    index=False
)

print("\nSaved results/algorithm_comparison.csv")
