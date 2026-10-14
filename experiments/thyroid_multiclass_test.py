
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

# 1. Load the dataset
df = pd.read_csv("dataset/thyroid.csv")

# 2. Remove identifier and dataset-source columns
df = df.drop(
    columns=["patient_id", "dataset_name"],
    errors="ignore"
)

# 3. Remove rows with missing target labels
df = df.dropna(subset=["class"])

# 4. Keep classes with at least 10 records
class_counts = df["class"].value_counts()
valid_classes = class_counts[class_counts >= 10].index
df = df[df["class"].isin(valid_classes)].copy()

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["class"].value_counts())

# 5. Separate input features and target
X = df.drop(columns=["class"])
y = df["class"]

# 6. Identify numerical and categorical columns
numeric_columns = X.select_dtypes(include=["number"]).columns
categorical_columns = X.select_dtypes(exclude=["number"]).columns

# 7. Prepare preprocessing
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

# 8. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 9. Train a class-weighted Random Forest
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

# 10. Evaluate the model
predictions = model.predict(X_test)

print("\n--- Multiclass Model Results ---")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 4)
)
print(
    "Macro F1-score:",
    round(f1_score(y_test, predictions, average="macro", zero_division=0), 4)
)

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))
