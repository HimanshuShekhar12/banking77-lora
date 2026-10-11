import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.prompt import build_messages

MODEL_PATH = (
    "/data/b23_himanshu_shekhar/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    local_files_only=True,
).to("cuda:1")

model.eval()

# Get all 77 intent categories from the training data
import pandas as pd

train_df = pd.read_csv("data/processed/train.csv")
categories = sorted(train_df["category"].unique())

text = "I am still waiting on my card."

messages = build_messages(text, categories)

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda:1")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        num_beams=1,
    )

# Decode only the newly generated tokens
new_tokens = output[0][inputs["input_ids"].shape[1]:]
prediction = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True,
).strip()

print("Customer message:", text)
print("Model prediction:", repr(prediction))
print("Ground truth for this example: card_arrival")
