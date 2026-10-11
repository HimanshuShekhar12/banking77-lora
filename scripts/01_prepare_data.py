from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
RAW_DIR = Path("data/raw/banking77")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
RANDOM_SEED = 42
VALIDATION_SIZE = 1000

# Load original training data
df = pd.read_csv(RAW_DIR / "train.csv")

# Basic validation
required_columns = {"text", "category"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Expected columns: {required_columns}")

if df[list(required_columns)].isnull().any().any():
    raise ValueError("Missing values found in the dataset.")

if df["category"].nunique() != 77:
    raise ValueError("Expected exactly 77 intent categories.")

# Stratified train/validation split
train_df, val_df = train_test_split(
    df,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_SEED,
    stratify=df["category"],
)

# Save processed splits
train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)

# Verify saved files
saved_train = pd.read_csv(PROCESSED_DIR / "train.csv")
saved_val = pd.read_csv(PROCESSED_DIR / "val.csv")

assert len(saved_train) == 9003
assert len(saved_val) == 1000
assert saved_train["category"].nunique() == 77
assert saved_val["category"].nunique() == 77

print("Data preparation complete!")
print(f"Training examples:   {len(saved_train)}")
print(f"Validation examples: {len(saved_val)}")
print(f"Intent categories:   {saved_train['category'].nunique()}")
print(f"Random seed:         {RANDOM_SEED}")
print(f"Saved to:            {PROCESSED_DIR.resolve()}")
