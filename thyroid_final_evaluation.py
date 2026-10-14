
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

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
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Load data
df = pd.read_csv("dataset/thyroid.csv")
df = df.drop(
    columns=["patient_id", "dataset_name"],
    errors="ignore"
)
df = df.dropna(subset=["class"])

# Keep the 13 classes used in our previous experiments
counts = df["class"].value_counts()
valid_classes = counts[counts >= 10].index
df = df[df["class"].isin(valid_classes)].copy()

X = df.drop(columns=["class"])
y = df["class"]

print("Classes evaluated:", y.nunique())
print("Records evaluated:", len(df))

# 2. Split into training (60%), validation (20%), test (20%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Test records:", len(X_test))

# 3. Preprocessing
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

# 4. Calculate class weights from TRAINING DATA ONLY
class_counts = y_train.value_counts()
n_samples = len(y_train)
n_classes = len(class_counts)

base_weights = {
    label: n_samples / (n_classes * count)
    for label, count in class_counts.items()
}

alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
validation_results = []
trained_models = {}

# 5. Select alpha using validation macro F1
for alpha in alphas:
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

    print(f"Training alpha={alpha}...")
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)

    score = f1_score(
        y_val,
        val_predictions,
        average="macro",
        zero_division=0
    )

    validation_results.append({
        "Alpha": alpha,
        "Validation Macro F1": score
    })
    trained_models[alpha] = model

validation_df = pd.DataFrame(validation_results)
print("\n--- Validation Results ---")
print(validation_df.round(4).to_string(index=False))

best_alpha = validation_df.loc[
    validation_df["Validation Macro F1"].idxmax(),
    "Alpha"
]
best_model = trained_models[best_alpha]

print("\nSelected alpha:", best_alpha)

# 6. Evaluate the selected model ONCE on the held-out test set
test_predictions = best_model.predict(X_test)

accuracy = accuracy_score(y_test, test_predictions)
balanced_accuracy = balanced_accuracy_score(
    y_test, test_predictions
)
macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0
)

print("\n--- Final Test Results ---")
print("Accuracy:", round(accuracy, 4))
print("Balanced accuracy:", round(balanced_accuracy, 4))
print("Macro F1:", round(macro_f1, 4))

print("\nClassification report:")
print(classification_report(
    y_test,
    test_predictions,
    zero_division=0
))

# 7. Save project outputs
os.makedirs("results", exist_ok=True)

validation_df.to_csv(
    "results/validation_weight_results.csv",
    index=False
)

pd.DataFrame([{
    "Model": "Logistic Regression",
    "Alpha": best_alpha,
    "Accuracy": accuracy,
    "Balanced Accuracy": balanced_accuracy,
    "Macro F1": macro_f1
}]).to_csv(
    "results/final_model_metrics.csv",
    index=False
)

joblib.dump(best_model, "results/thyroid_best_model.joblib")

# 8. Save confusion matrix
fig, ax = plt.subplots(figsize=(14, 12))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_predictions,
    labels=best_model.classes_,
    xticks_rotation=90,
    cmap="Blues",
    colorbar=False,
    ax=ax
)

ax.set_title("Thyroid Classification - Final Test Confusion Matrix")
fig.tight_layout()
fig.savefig(
    "results/confusion_matrix_final.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close(fig)

print("\nSaved final metrics, selected model, and confusion matrix")
