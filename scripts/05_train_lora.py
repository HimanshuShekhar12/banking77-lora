import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

from src.prompt import build_messages, build_training_messages


MODEL_PATH = (
    "/data/b23_himanshu_shekhar/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/val.csv"
OUTPUT_DIR = "results/lora"
MAX_LENGTH = 768


class BankingDataset(Dataset):
    def __init__(self, csv_path, tokenizer, categories):
        df = pd.read_csv(csv_path)
        self.examples = []

        for row in df.itertuples(index=False):
            text = str(row.text)
            answer = str(row.category)

            # Full conversation: prompt + correct answer.
            full_messages = build_training_messages(
                text, categories, answer
            )

            full_ids = tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
                truncation=True,
                max_length=MAX_LENGTH,
            )

            # Prompt only: no answer included.
            prompt_messages = build_messages(text, categories)

            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                truncation=True,
                max_length=MAX_LENGTH,
            )

            # The prompt tokens must be a prefix of the full conversation.
            if full_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError(
                    "Prompt/full-conversation token mismatch. "
                    "Check the chat template before training."
                )

            labels = full_ids.copy()

            # Ignore prompt tokens when calculating the loss.
            labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

            # Don't train on examples whose answer was truncated away.
            if all(token == -100 for token in labels):
                raise ValueError(
                    f"No answer tokens left for example: {text}"
                )

            self.examples.append({
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            })

        print(f"Loaded {len(self.examples)} examples from {csv_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(
            [
                {
                    "input_ids": item["input_ids"],
                    "attention_mask": item["attention_mask"],
                }
                for item in features
            ],
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        padded_labels = []

        for item in features:
            labels = item["labels"]
            padding_length = max_len - len(labels)
            padded_labels.append(
                labels + [-100] * padding_length
            )

        batch["labels"] = torch.tensor(
            padded_labels, dtype=torch.long
        )
        return batch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check your environment.")

    print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    categories = sorted(train_df["category"].unique().tolist())

    assert len(categories) == 77, (
        f"Expected 77 intents, found {len(categories)}"
    )
    assert set(val_df["category"].unique()) == set(categories), (
        "Training and validation intent labels differ."
    )

    print("Number of intents:", len(categories))
    print("Training examples:", len(train_df))
    print("Validation examples:", len(val_df))

    train_dataset = BankingDataset(
        TRAIN_PATH, tokenizer, categories
    )
    val_dataset = BankingDataset(
        VAL_PATH, tokenizer, categories
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=20,
        report_to="none",
        seed=42,
        data_seed=42,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollator(tokenizer),
        processing_class=tokenizer,
    )

    train_result = trainer.train()

    # Save the best selected adapter and tokenizer.
    trainer.save_model(f"{OUTPUT_DIR}/best_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/best_adapter")

    summary = {
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "lora_rank": 16,
        "lora_alpha": 32,
        "epochs_requested": 2,
        "train_examples": len(train_dataset),
        "validation_examples": len(val_dataset),
        "train_runtime_seconds": train_result.metrics.get(
            "train_runtime"
        ),
        "train_metrics": train_result.metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "categories": categories,
    }

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(f"{OUTPUT_DIR}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
