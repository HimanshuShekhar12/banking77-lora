import pandas as pd
from transformers import AutoTokenizer

from src.prompt import build_messages, build_training_messages

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

train = pd.read_csv("data/processed/train.csv")
categories = sorted(train["category"].unique())

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = train.iloc[0]["text"]
answer = train.iloc[0]["category"]

messages = build_messages(text, categories)

# Input prompt: no correct answer included
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# Complete conversation: used to inspect training format
full_messages = build_training_messages(text, categories, answer)
full_text = tokenizer.apply_chat_template(
    full_messages,
    tokenize=False,
    add_generation_prompt=False,
)

prompt_tokens = tokenizer(
    prompt, add_special_tokens=False
)["input_ids"]

full_tokens = tokenizer(
    full_text, add_special_tokens=False
)["input_ids"]

print("Model:", MODEL_NAME)
print("Number of intents:", len(categories))
print("Sample message:", text)
print("Expected target:", answer)
print("\nPrompt preview:\n")
print(prompt[:1500])
print("\nPrompt token count:", len(prompt_tokens))
print("Full conversation token count:", len(full_tokens))
print("Tokenizer loaded successfully.")
