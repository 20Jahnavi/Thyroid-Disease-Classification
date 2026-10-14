
import pandas as pd

df = pd.read_csv("dataset/thyroid.csv")

print("Dataset shape:", df.shape)

print("\nTarget label counts:")
print(df["class"].value_counts(dropna=False))

print("\nMissing values in each column:")
print(df.isna().sum().sort_values(ascending=False).to_string())

print("\nData types:")
print(df.dtypes.to_string())

print("\nFirst 5 rows:")
print(df.head().to_string())

# Check whether duplicate rows exist
print("\nExact duplicate rows:", df.duplicated().sum())

# Check whether the target labels have whitespace
labels = df["class"].dropna().astype(str)
print("\nLabels with leading/trailing spaces:")
print(labels[labels != labels.str.strip()].unique())

# Inspect columns that may be related to the target or dataset source
print("\nColumns:")
print(df.columns.tolist())
