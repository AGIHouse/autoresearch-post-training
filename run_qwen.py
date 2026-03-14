"""
Downloads Qwen/Qwen3.5-0.8B-Base from Hugging Face and runs inference
on a small dataset of text-completion prompts.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MAX_NEW_TOKENS = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 1. Build a simple in-memory dataset
# ---------------------------------------------------------------------------
prompts = [
    "The capital of France is",
    "Water boils at 100 degrees Celsius, which means",
    "In machine learning, a neural network is",
    "The speed of light is approximately 299,792 kilometers per second, so",
    "The first president of the United States was",
]
dataset = Dataset.from_dict({"prompt": prompts})

# ---------------------------------------------------------------------------
# 2. Load tokenizer & model
# ---------------------------------------------------------------------------
print(f"Loading tokenizer for {MODEL_ID} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print(f"Loading model onto {DEVICE} …")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print("Model loaded.\n")

# ---------------------------------------------------------------------------
# 3. Run inference
# ---------------------------------------------------------------------------
print("=" * 70)
print(f"Running inference on {len(dataset)} examples  (max_new_tokens={MAX_NEW_TOKENS})")
print("=" * 70)

for i, example in enumerate(dataset):
    prompt = example["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,          # greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"\n[{i+1}] PROMPT   : {prompt}")
    print(f"    COMPLETION: {completion.strip()}")

print("\n" + "=" * 70)
print("Done.")
