from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
from tqdm import tqdm

# ---- Step 1: Load the model and tokenizer ----
model_name = "utter-project/EuroLLM-9B-Instruct"  # or "utter-project/EuroLLM-1.7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

translator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.2,
)

# ---- Step 2: Read your CSV ----
csv_path = "../../MathNeuro/data/race.csv"  # path to your file
df = pd.read_csv(csv_path)

# ---- Step 3: Prepare output column ----
df["translation_de"] = ""

# ---- Step 4: Iterate and translate each row ----
for i, text in tqdm(enumerate(df["article"]), total=len(df), desc="Translating"):
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        df.at[i, "translation_de"] = ""
        continue

    prompt = f"Translate the following English text into German:\n\n{text.strip()}\n\nGerman translation:"
    output_path = "translated.csv"
    try:
        output = translator(prompt)[0]["generated_text"]
        translation = output.split("German translation:")[-1].strip()
        df.at[i, "translation_de"] = translation

        df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Error translating row {i}: {e}")
        df.at[i, "translation_de"] = ""

# ---- Step 5: Save results ----
output_path = "C:/Users/timse/Documents/Lab_test/LabTest/translated.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Translations saved to {output_path}")
