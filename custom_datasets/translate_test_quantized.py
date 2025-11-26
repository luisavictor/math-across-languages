from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# ---- Step 1: Load the (quantized) model and tokenizer ----
model_name = "utter-project/EuroLLM-9B-Instruct"  # 9B model

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16  # use torch.float16 if bfloat16 isn't supported
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,   # quantized to 4-bit
    device_map="auto",
    trust_remote_code=True
)

translator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.2,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# ---- Step 2: Read your CSV ----
csv_path = "../MathNeuro/data/race.csv"
df = pd.read_csv(csv_path)

# ---- Step 3: Prepare output column ----
if "translation_de" not in df.columns:
    df["translation_de"] = ""

# ---- Step 4: Iterate, translate, and SAVE AFTER EACH ROW ----
final_output_path = Path("C:/Users/timse/Documents/Lab_test/LabTest/translated.csv")
final_output_path.parent.mkdir(parents=True, exist_ok=True)

for i, text in tqdm(enumerate(df["article"]), total=len(df), desc="Translating"):
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        df.at[i, "translation_de"] = ""
        # save progress every iteration
        df.to_csv(final_output_path, index=False)
        continue

    prompt = (
        "Translate the following English text into German:\n\n"
        f"{text.strip()}\n\nGerman translation:"
    )

    try:
        out = translator(prompt, return_full_text=True)[0]["generated_text"]
        translation = out.split("German translation:")[-1].strip()
        df.at[i, "translation_de"] = translation
    except Exception as e:
        print(f"Error translating row {i}: {e}")
        df.at[i, "translation_de"] = ""

    # ✅ Save the entire dataframe on every iteration
    df.to_csv(final_output_path, index=False)

print(f"\n✅ Translations saved to {final_output_path}")
