# Banking77 Intent Classification with LoRA and QLoRA

Compare parameter-efficient fine-tuning methods for classifying banking customer messages into 77 customer-support intents using Qwen2.5-1.5B-Instruct.

## Project Overview

This project fine-tunes a small instruction-tuned language model to map customer messages to one of 77 Banking77 intent labels. It compares the base model, LoRA fine-tuning, and 4-bit QLoRA fine-tuning.

### Key Results

| Model | Test Accuracy | Test Macro-F1 | Invalid Outputs |
|---|---:|---:|---:|
| Qwen2.5-1.5B-Instruct (Base) | 45.29% | 41.62% | 104 |
| Qwen2.5-1.5B-Instruct + LoRA (r=16) | **93.05%** | **93.05%** | 2 |
| Qwen2.5-1.5B-Instruct + QLoRA (r=16) | 92.95% | 92.97% | 2 |

All test metrics are calculated on the official Banking77 test set containing 3,080 examples. The validation set was used for model selection.

## Efficiency Benchmark

Inference benchmark on an NVIDIA RTX A5000 GPU, using 5 warm-up examples and 95 measured examples per method.

| Metric | LoRA | QLoRA |
|---|---:|---:|
| Model Loading Time | 1.41 s | 2.29 s |
| Mean Inference Latency | 145.85 ms | 187.08 ms |
| Median Inference Latency | 150.94 ms | 195.81 ms |
| Throughput | 6.86 examples/s | 5.35 examples/s |
| Peak Allocated GPU Memory | 3.03 GB | 1.23 GB |
| Peak Reserved GPU Memory | 3.18 GB | 3.05 GB |

In this inference benchmark, QLoRA reduced peak allocated GPU memory by approximately 59.4%, while LoRA achieved higher throughput. These are inference measurements, not training-memory measurements.

## Dataset

- **Dataset:** Banking77
- **Number of intents:** 77
- **Training examples:** 9,003
- **Validation examples:** 1,000
- **Official test examples:** 3,080

The official test set is kept separate from training and validation.

## Models and Methods

### LoRA
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Training epochs: 2
- Learning rate: `2e-4`
- Trainable parameters: 18,464,768
- Total parameters: 1,562,179,072
- Trainable parameter fraction: 1.182%

### QLoRA
- 4-bit NF4 quantization
- Double quantization enabled
- FP16 compute
- LoRA rank: 16
- LoRA alpha: 32
- Training epochs: 2

## Project Structure

```text
banking77-lora/
├── configs/
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 03_single_prediction.py
│   ├── 04_eval_baseline.py
│   ├── 05_train_lora.py
│   ├── 05_train_lora_r8.py
│   ├── 06_eval_lora.py
│   ├── 06_eval_lora_r8.py
│   ├── 06_eval_qlora.py
│   ├── 07_eval_lora_test.py
│   ├── 07_eval_qlora_test.py
│   ├── 08_error_analysis.py
│   ├── 09_train_qlora.py
│   └── 10_benchmark_efficiency.py
├── src/
│   └── prompt.py
├── results/
│   └── efficiency_benchmark.json
├── requirements.txt
└── README.md
```

Dataset files, generated predictions, model adapters, and training checkpoints are excluded from version control.

## Setup

1. Clone this repository.
2. Create and activate a Python environment.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Download the Banking77 dataset and arrange the CSV files according to the paths expected by the scripts.
5. Make the pretrained Qwen2.5-1.5B-Instruct model available locally, or update the model path in the scripts.

The scripts were developed in a CUDA-enabled environment. GPU support and compatible PyTorch, Transformers, PEFT, and bitsandbytes versions are required for the full training and benchmarking workflow.

## Reproducing Experiments

Run commands from the repository root. Example commands:

```bash
python scripts/01_prepare_data.py
python scripts/04_eval_baseline.py
python scripts/05_train_lora.py
python scripts/09_train_qlora.py
python scripts/10_benchmark_efficiency.py
```

Check each script's configuration and expected input paths before running it. Training and evaluation scripts may require substantial GPU memory and compute time.

## Error Analysis

The LoRA test evaluation produced 214 incorrect predictions out of 3,080 examples. Common confusions included closely related intents such as card-payment recognition, card arrival versus delivery estimates, and transfer status categories.

## Limitations

- Results depend on the prompt, model version, hardware, and software environment.
- The efficiency benchmark covers 95 measured examples per method after warm-up; results may vary across runs.
- Inference memory measurements should not be interpreted as training-memory measurements.
- Intent classification errors can occur between semantically similar categories.

## Future Improvements

- Benchmark repeated runs and report variability.
- Evaluate batch inference and longer workloads.
- Investigate difficult intent pairs using error analysis.
- Improve experiment configuration and reproducibility.

## Acknowledgements

- Banking77 dataset
- Qwen2.5-1.5B-Instruct
- PEFT and bitsandbytes
