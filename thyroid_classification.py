
# Thyroid Disease Classification using Pattern Recognition

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

DATA_PATH = "dataset/thyroid.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

data = pd.read_csv(DATA_PATH)

print("=" * 60)
print("THYROID DISEASE CLASSIFICATION")
print("=" * 60)

print("\nDataset shape:", data.shape)

TARGET = "class"

if TARGET not in data.columns:
    raise ValueError("Target column 'class' not found in dataset.")

# Remove rows without a target label
data = data.dropna(subset=[TARGET]).copy()

# Remove identifier and dataset-source columns
data = data.drop(
    columns=["patient_id", "dataset_name"],
    errors="ignore"
)

# --------------------------------------------------
# 2. Examine target distribution
# --------------------------------------------------

print("\nOriginal target distribution:")
print(data[TARGET].value_counts())

# A class with only one example cannot be placed in
# both training and testing sets using stratification.
counts = data[TARGET].value_counts()
rare_classes = counts[counts < 2].index.tolist()

if rare_classes:
    print("\nExcluding classes with fewer than 2 records:")
    print(rare_classes)
    data = data[~data[TARGET].isin(rare_classes)].copy()

print("\nRecords used:", len(data))
print("Number of classes:", data[TARGET].nunique())

# --------------------------------------------------
# 3. Separate features and target
# --------------------------------------------------

X = data.drop(columns=[TARGET])
y = data[TARGET].astype(str)

# Identify numerical and categorical attributes
numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumerical features:", len(numeric_columns))
print("Categorical features:", len(categorical_columns))

# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# --------------------------------------------------
# 5. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 6. Define classification models
# --------------------------------------------------

models = {
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "Naive Bayes": GaussianNB(),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

# --------------------------------------------------
# 7. Train and evaluate models
# --------------------------------------------------

results = []
trained_models = {}

for name, classifier in models.items():

    print("\n" + "-" * 60)
    print("Training:", name)

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_acc = balanced_accuracy_score(
        y_test, predictions
    )
    precision = precision_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )
    recall = recall_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )
    f1 = f1_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_acc,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1
    })

    trained_models[name] = model

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Macro Precision:   {precision:.4f}")
    print(f"Macro Recall:      {recall:.4f}")
    print(f"Macro F1-score:    {f1:.4f}")

# --------------------------------------------------
# 8. Compare model performance
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(
    index=False,
    formatters={
        col: "{:.4f}".format
        for col in results_df.columns
        if col != "Model"
    }
))

results_df.to_csv(
    os.path.join(RESULTS_DIR, "model_metrics.csv"),
    index=False
)

# --------------------------------------------------
# 9. Plot model comparison
# --------------------------------------------------

plot_data = results_df.set_index("Model")[
    ["Accuracy", "Balanced Accuracy", "Macro F1"]
]

ax = plot_data.plot(
    kind="bar",
    figsize=(11, 6)
)

ax.set_title("Thyroid Classifier Performance Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Classification Model")
ax.set_ylim(0, 1)
plt.xticks(rotation=20, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "model_comparison.png"),
    dpi=300
)
plt.close()

# --------------------------------------------------
# 10. Best model by macro F1-score
# --------------------------------------------------

best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

best_predictions = best_model.predict(X_test)

print("\n" + "=" * 60)
print("BEST MODEL:", best_name)
print("=" * 60)

print("\nClassification report:")
print(classification_report(
    y_test,
    best_predictions,
    zero_division=0
))

# --------------------------------------------------
# 11. Confusion matrix
# --------------------------------------------------

labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=labels
)

plt.figure(figsize=(14, 11))

sns.heatmap(
    cm,
    cmap="Blues",
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels
)

plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "confusion_matrix.png"),
    dpi=300
)
plt.close()

# --------------------------------------------------
# 12. Save the best trained model
# --------------------------------------------------

import joblib

joblib.dump(
    best_model,
    "thyroid_best_model.joblib"
)

print("\nFiles generated:")
print("- results/model_comparison.png")
print("- results/confusion_matrix.png")
print("- results/model_metrics.csv")
print("- thyroid_best_model.joblib")

print("\nProject execution completed!")
