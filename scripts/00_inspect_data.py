from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/banking77")

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

for name, df in [("TRAIN", train), ("TEST", test)]:
    print(f"\n{'=' * 15} {name} {'=' * 15}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    print("\nMissing values:")
    print(df.isnull().sum())

    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print(f"Duplicate texts: {df['text'].duplicated().sum()}")
    print(f"Unique categories: {df['category'].nunique()}")

    print("\nTop 5 categories:")
    print(df["category"].value_counts().head())

    print("\nBottom 5 categories:")
    print(df["category"].value_counts().tail())

# Cross-dataset checks
train_categories = set(train["category"].unique())
test_categories = set(test["category"].unique())

print("\n" + "=" * 15 + " CROSS-DATASET " + "=" * 15)
print("Train rows:", len(train))
print("Test rows:", len(test))
print("Train categories:", len(train_categories))
print("Test categories:", len(test_categories))
print("Categories missing from test:", train_categories - test_categories)

train_texts = set(train["text"].dropna())
test_texts = set(test["text"].dropna())
print("Texts appearing in both train and test:", len(train_texts & test_texts))

print("\nSample training examples:")
print(train.head(5).to_string(index=False))
