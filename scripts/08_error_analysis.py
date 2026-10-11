import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PREDICTIONS_PATH = Path("results/lora/test_lora_predictions.jsonl")
TRAIN_PATH = Path("data/processed/train.csv")
OUTPUT_DIR = Path("results/lora/error_analysis")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load predictions made earlier; no model inference needed.
    df = pd.read_json(PREDICTIONS_PATH, lines=True)
    categories = sorted(
        pd.read_csv(TRAIN_PATH)["category"].unique().tolist()
    )

    y_true = df["true_label"].astype(str)
    y_pred = df["prediction"].astype(str)

    invalid_mask = ~y_pred.isin(categories)

    summary = {
        "num_examples": len(df),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=categories,
                average="macro",
                zero_division=0,
            )
        ),
        "correct_predictions": int((y_true == y_pred).sum()),
        "incorrect_predictions": int((y_true != y_pred).sum()),
        "invalid_outputs": int(invalid_mask.sum()),
    }

    # Per-intent precision, recall, F1 and support.
    report = classification_report(
        y_true,
        y_pred,
        labels=categories,
        output_dict=True,
        zero_division=0,
    )

    per_class = pd.DataFrame(
        [
            {
                "intent": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            }
            for label in categories
        ]
    ).sort_values("f1")

    per_class.to_csv(OUTPUT_DIR / "per_class_metrics.csv", index=False)

    # Save every misclassified example for manual inspection.
    errors = df.loc[y_true != y_pred].copy()
    errors["is_invalid_output"] = invalid_mask[y_true != y_pred].values
    errors.to_csv(OUTPUT_DIR / "misclassified_examples.csv", index=False)

    # Include invalid outputs as a separate prediction column.
    pred_for_matrix = y_pred.where(y_pred.isin(categories), "INVALID_OUTPUT")
    matrix_labels = categories + ["INVALID_OUTPUT"]

    cm = confusion_matrix(
        y_true,
        pred_for_matrix,
        labels=matrix_labels,
    )

    matrix_df = pd.DataFrame(
        cm,
        index=matrix_labels,
        columns=matrix_labels,
    )
    matrix_df.index.name = "true_intent"
    matrix_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    # Most frequent wrong intent pairs.
    wrong = df.loc[y_true != y_pred].copy()
    wrong_pairs = (
        wrong.groupby(["true_label", "prediction"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    wrong_pairs.to_csv(OUTPUT_DIR / "top_confusions.csv", index=False)

    with (OUTPUT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== ERROR ANALYSIS SUMMARY ===")
    print(json.dumps(summary, indent=2))

    print("\n=== 10 LOWEST-F1 INTENTS ===")
    print(per_class.head(10).to_string(index=False))

    print("\n=== MOST COMMON CONFUSIONS ===")
    print(wrong_pairs.head(15).to_string(index=False))

    print("\n=== SAMPLE ERRORS ===")
    for row in errors.head(10).itertuples(index=False):
        print(f"\nText: {row.text}")
        print(f"True: {row.true_label}")
        print(f"Predicted: {row.prediction}")

    print(f"\nFiles saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
