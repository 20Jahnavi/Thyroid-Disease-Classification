
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
    f1_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

# 1. Load the dataset
data = pd.read_csv("dataset/thyroid.csv")
data = data.dropna(subset=["class"]).copy()

# 2. Remove identifiers and source information
X = data.drop(
    columns=["class", "patient_id", "dataset_name"],
    errors="ignore"
)

# 3. Create a binary target
y = (data["class"] != "negative").astype(int)

print("Target distribution:")
print(y.value_counts().rename({
    0: "negative",
    1: "thyroid-related"
}))

# 4. Identify feature types
numeric_cols = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_cols = X.select_dtypes(
    exclude="number"
).columns.tolist()

# 5. Preprocess features
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

# 6. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 7. Compare baseline and classification models
models = {
    "Majority baseline": DummyClassifier(
        strategy="most_frequent"
    ),
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

for name, classifier in models.items():
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("Accuracy:", round(
        accuracy_score(y_test, predictions), 4
    ))
    print("Balanced accuracy:", round(
        balanced_accuracy_score(y_test, predictions), 4
    ))
    print("Macro F1:", round(
        f1_score(y_test, predictions, average="macro"), 4
    ))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["negative", "thyroid-related"],
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
