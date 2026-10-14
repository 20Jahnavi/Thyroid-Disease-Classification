
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. Load and clean data
df = pd.read_csv("dataset/thyroid.csv")
df = df.drop(
    columns=["patient_id", "dataset_name"],
    errors="ignore"
)
df = df.dropna(subset=["class"])

# Keep classes with at least 10 records
counts = df["class"].value_counts()
valid_classes = counts[counts >= 10].index
df = df[df["class"].isin(valid_classes)].copy()

X = df.drop(columns=["class"])
y = df["class"]

# 2. Identify column types
numeric_columns = X.select_dtypes(include=["number"]).columns
categorical_columns = X.select_dtypes(exclude=["number"]).columns

# 3. Preprocess the features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# 4. Split BEFORE oversampling
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Oversample minority classes in training data only.
# Raise each smaller class to 1,000 training examples.
train_counts = y_train.value_counts()
sampling_strategy = {
    label: 1000
    for label, count in train_counts.items()
    if count < 1000
}

model = ImbPipeline([
    ("preprocessor", preprocessor),
    ("oversampler", RandomOverSampler(random_state=42,
                                      sampling_strategy=sampling_strategy)),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

# 6. Train and evaluate
print("Training model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n--- Oversampling Model Results ---")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 4)
)
print(
    "Macro F1-score:",
    round(f1_score(y_test, predictions, average="macro",
                   zero_division=0), 4)
)

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))
