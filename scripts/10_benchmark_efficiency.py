import gc
import json
import time
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.prompt import build_messages

MODEL_PATH = (
    "/data/b23_himanshu_shekhar/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
DATA_PATH = "data/raw/banking77/test.csv"
LORA_PATH = "results/lora/best_adapter"
QLORA_PATH = "results/qlora/best_adapter"
OUT_PATH = Path("results/efficiency_benchmark.json")

# Same examples and settings for both models.
N_EXAMPLES = 100
WARMUP_EXAMPLES = 5
MAX_NEW_TOKENS = 20


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def load_model(method, tokenizer):
    start = time.perf_counter()

    if method == "LoRA":
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map={"": 0},
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(base, LORA_PATH)

    else:
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
        model = PeftModel.from_pretrained(base, QLORA_PATH)

    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    return model, load_seconds


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    tokenizer = AutoTokenizer.from_pretrained(
        LORA_PATH, local_files_only=True
    )
    df = pd.read_csv(DATA_PATH).head(N_EXAMPLES)
    categories = sorted(
        pd.read_csv("data/processed/train.csv")["category"].unique()
    )

    messages_list = [
        build_messages(str(row.text), categories)
        for row in df.itertuples(index=False)
    ]

    results = []

    for method in ["LoRA", "QLoRA"]:
        print(f"\nBenchmarking {method}...", flush=True)
        cleanup()

        # Measure peak allocated/reserved memory from a clean baseline.
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        model, load_seconds = load_model(method, tokenizer)

        # Warm up first; exclude warmup examples from latency results.
        for messages in messages_list[:WARMUP_EXAMPLES]:
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda:0")
            with torch.inference_mode():
                model.generate(
                    inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        torch.cuda.synchronize()

        latencies = []
        input_tokens = 0
        generated_tokens = 0
        torch.cuda.reset_peak_memory_stats()

        for i, messages in enumerate(
            messages_list[WARMUP_EXAMPLES:], start=1
        ):
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda:0")

            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(
                    inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)
            input_tokens += inputs.shape[1]
            generated_tokens += output.shape[1] - inputs.shape[1]

            if i % 20 == 0:
                print(
                    f"{method}: measured {i}/"
                    f"{len(messages_list) - WARMUP_EXAMPLES}",
                    flush=True,
                )

        total_seconds = sum(latencies)
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()

        result = {
            "method": method,
            "gpu": torch.cuda.get_device_name(0),
            "examples_measured": len(latencies),
            "warmup_examples": WARMUP_EXAMPLES,
            "model_load_seconds": load_seconds,
            "mean_latency_seconds": sum(latencies) / len(latencies),
            "median_latency_seconds": sorted(latencies)[len(latencies) // 2],
            "throughput_examples_per_second": len(latencies) / total_seconds,
            "generated_tokens_per_second": generated_tokens / total_seconds,
            "peak_allocated_gb": peak_allocated / (1024**3),
            "peak_reserved_gb": peak_reserved / (1024**3),
            "baseline_allocated_gb": baseline_allocated / (1024**3),
            "baseline_reserved_gb": baseline_reserved / (1024**3),
            "input_tokens_measured": input_tokens,
            "generated_tokens_measured": generated_tokens,
        }
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

        del model
        cleanup()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved benchmark results to {OUT_PATH}")


if __name__ == "__main__":
    main()
