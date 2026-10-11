import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompt import build_messages

MODEL_PATH = (
    "/data/b23_himanshu_shekhar/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)

DATA_DIR = Path("data/processed")
RAW_DIR = Path("data/raw/banking77")
RESULTS_DIR = Path("results/baseline")
DEVICE = "cuda:1"
MAX_NEW_TOKENS = 20


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        local_files_only=True,
    ).to(DEVICE)

    model.eval()
    return tokenizer, model


def predict(text, categories, tokenizer, model):
    messages = build_messages(text, categories)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
        )

    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    prediction = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    return prediction


def evaluate_split(split, categories, tokenizer, model):
    if split == "val":
        df = pd.read_csv(DATA_DIR / "val.csv")
    elif split == "test":
        # Official test file remains unchanged.
        df = pd.read_csv(RAW_DIR / "test.csv")
    else:
        raise ValueError(f"Unknown split: {split}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / f"{split}_predictions.jsonl"

    # Resume: reuse previously completed predictions.
    predictions_by_index = {}

    if output_file.exists():
        with output_file.open() as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    predictions_by_index[record["index"]] = record

    remaining = len(df) - len(predictions_by_index)
    print(f"\nEvaluating {split}: {len(df)} examples")
    print(f"Already completed: {len(predictions_by_index)}")
    print(f"Remaining: {remaining}")

    with output_file.open("a") as f:
        for i, row in df.iterrows():
            if i in predictions_by_index:
                continue

            prediction = predict(
                row["text"], categories, tokenizer, model
            )

            record = {
                "index": int(i),
                "text": row["text"],
                "true_label": row["category"],
                "prediction": prediction,
            }

            f.write(json.dumps(record) + "\n")
            f.flush()

            predictions_by_index[i] = record

            completed = len(predictions_by_index)
            if completed % 25 == 0 or completed == len(df):
                print(f"Progress: {completed}/{len(df)}")

    records = [predictions_by_index[i] for i in range(len(df))]
    y_true = [r["true_label"] for r in records]
    y_pred = [r["prediction"] for r in records]

    valid_categories = set(categories)
    invalid_count = sum(p not in valid_categories for p in y_pred)

    metrics = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "split": split,
        "num_examples": len(df),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, labels=categories,
            average="macro", zero_division=0,
        ),
        "invalid_output_rate": invalid_count / len(df),
        "invalid_output_count": invalid_count,
    }

    metrics_file = RESULTS_DIR / f"{split}_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2))

    print(f"\nResults for {split}:")
    print(json.dumps(metrics, indent=2))
    print(f"Predictions saved to: {output_file}")
    print(f"Metrics saved to: {metrics_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["val", "test"],
        default=["val", "test"],
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this environment.")

    train = pd.read_csv(DATA_DIR / "train.csv")
    categories = sorted(train["category"].unique())

    print("Device:", DEVICE)
    print("GPU:", torch.cuda.get_device_name(1))
    print("Number of intents:", len(categories))

    tokenizer, model = load_model()

    for split in args.splits:
        evaluate_split(split, categories, tokenizer, model)


if __name__ == "__main__":
    main()
