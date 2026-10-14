
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve
)

# 1. Load data
data = pd.read_csv("dataset/thyroid.csv")
data = data.dropna(subset=["class"]).copy()

X = data.drop(
    columns=["class", "patient_id", "dataset_name"],
    errors="ignore"
)

y = (data["class"] != "negative").astype(int)

# 2. Split into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val
)

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Test records:", len(X_test))

# 3. Preprocess the features
numeric_cols = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_cols = X.select_dtypes(
    exclude="number"
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

# 4. Train Logistic Regression
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1500,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# 5. Select a threshold using validation data only
val_probabilities = model.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_val, val_probabilities
)

# Calculate F1 for every available threshold
f1_values = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-12)
)

best_index = np.argmax(f1_values)
best_threshold = thresholds[best_index]

print("\nValidation-selected threshold:", round(
    float(best_threshold), 4
))
print("Validation positive-class F1:", round(
    float(f1_values[best_index]), 4
))

# 6. Evaluate once on the untouched test set
test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\nTEST RESULTS")
print("=" * 45)

print("Accuracy:", round(
    accuracy_score(y_test, test_predictions), 4
))
print("Balanced accuracy:", round(
    balanced_accuracy_score(y_test, test_predictions), 4
))
print("Positive-class precision:", round(
    precision_score(y_test, test_predictions, zero_division=0), 4
))
print("Positive-class recall:", round(
    recall_score(y_test, test_predictions, zero_division=0), 4
))
print("Positive-class F1:", round(
    f1_score(y_test, test_predictions, zero_division=0), 4
))

print("\nClassification report:")
print(classification_report(
    y_test,
    test_predictions,
    labels=[0, 1],
    target_names=["negative", "thyroid-related"],
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(y_test, test_predictions, labels=[0, 1]))
