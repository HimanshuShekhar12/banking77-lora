import json
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

from src.prompt import build_messages

MODEL_PATH = (
    "/data/b23_himanshu_shekhar/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
ADAPTER_PATH = "results/lora/best_adapter"
DATA_PATH = "data/raw/banking77/test.csv"
OUTPUT_DIR = Path("results/lora")

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH, local_files_only=True
    )

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.to(device)
    model.eval()

    df = pd.read_csv(DATA_PATH)
    categories = sorted(
        pd.read_csv("data/processed/train.csv")["category"].unique()
    )

    predictions = []
    output_path = OUTPUT_DIR / "test_lora_predictions.jsonl"

    with output_path.open("w") as f:
        for i, row in enumerate(df.itertuples(index=False), start=1):
            messages = build_messages(str(row.text), categories)
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

            truth = str(row.category)
            valid = prediction in categories
            predictions.append((truth, prediction))

            f.write(json.dumps({
                "text": str(row.text),
                "true_label": truth,
                "prediction": prediction,
                "valid_output": valid,
            }) + "\n")

            if i % 50 == 0:
                print(f"Evaluated {i}/{len(df)} examples")

    y_true = [x[0] for x in predictions]
    y_pred = [x[1] for x in predictions]

    metrics = {
        "model": "Qwen2.5-1.5B-Instruct + LoRA",
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

    with (OUTPUT_DIR / "test_lora_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print("\nLoRA validation metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
