import json
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, f1_score

from src.prompt import build_messages

MODEL_PATH = (
    "/data/b23_himanshu_shekhar/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
ADAPTER_PATH = "results/qlora/best_adapter"
DATA_PATH = "data/raw/banking77/test.csv"
OUTPUT_DIR = Path("results/qlora")

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH, local_files_only=True
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map={"": 0},
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.to(device)
    model.eval()

    df = pd.read_csv(DATA_PATH)
    categories = sorted(
        pd.read_csv("data/processed/train.csv")["category"].unique()
    )

    output_path = OUTPUT_DIR / "test_qlora_predictions.jsonl"

    # Load and verify predictions already saved.
    saved = []
    if output_path.exists():
        with output_path.open() as f:
            saved = [json.loads(line) for line in f if line.strip()]

    if len(saved) > len(df):
        raise RuntimeError("Saved predictions exceed test dataset size.")

    for i, record in enumerate(saved):
        row = df.iloc[i]
        if (
            record.get("text") != str(row["text"])
            or record.get("true_label") != str(row["category"])
        ):
            raise RuntimeError(
                f"Saved prediction mismatch at row {i + 1}. "
                "No new predictions were generated."
            )

    predictions = [
        (record["true_label"], record["prediction"])
        for record in saved
    ]

    print(f"Already saved: {len(saved)}/{len(df)}")
    print(f"Remaining: {len(df) - len(saved)}")

    # Append only the remaining examples.
    with output_path.open("a") as f:
        for i in range(len(saved), len(df)):
            row = df.iloc[i]
            messages = build_messages(str(row["text"]), categories)
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)

            with torch.inference_mode():
                output = model.generate(
                    inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            new_tokens = output[0, inputs.shape[1]:]
            prediction = tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()

            truth = str(row["category"])
            valid = prediction in categories
            predictions.append((truth, prediction))

            f.write(json.dumps({
                "text": str(row["text"]),
                "true_label": truth,
                "prediction": prediction,
                "valid_output": valid,
            }) + "\\n")
            f.flush()

            completed = i + 1
            if completed % 50 == 0 or completed == len(df):
                print(f"Evaluated {completed}/{len(df)} examples")

    y_true = [x[0] for x in predictions]
    y_pred = [x[1] for x in predictions]

    metrics = {
        "model": "Qwen2.5-1.5B-Instruct + QLoRA",
        "split": "test",
        "num_examples": len(df),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, labels=categories,
            average="macro", zero_division=0
        ),
        "invalid_output_count": sum(p not in categories for p in y_pred),
        "invalid_output_rate": sum(p not in categories for p in y_pred) / len(y_pred),
    }

    with (OUTPUT_DIR / "test_qlora_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print("\nLoRA validation metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
