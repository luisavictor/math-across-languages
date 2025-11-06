from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# ---- Configuration ----
TARGET_LANGUAGES = {
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "nl": "Dutch",
    "pt": "Portuguese"
}

MODEL_NAME = "utter-project/EuroLLM-9B-Instruct"
INPUT_CSV = "MathNeuro/data/race.csv"
OUTPUT_DIR = "C:/Users/timse/Documents/Lab_test/LabTest"

# ---- Step 1: Load quantized model ----
print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

translator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# ---- Step 2: Translation function ----
def translate_text(text, target_language):
    """Translate text to target language"""
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return ""
    
    prompt = f"Translate the following English text into {target_language}. Do not add any text or advices more to it and just output the translated text only:\n\n{text.strip()}\n\n{target_language}:"
    
    try:
        out = translator(prompt, return_full_text=False)[0]["generated_text"]
        return out.strip()
    except Exception as e:
        print(f"Error during translation: {e}")
        return ""

# ---- Step 3: Process each language ----
for lang_code, lang_name in TARGET_LANGUAGES.items():
    print(f"\n{'='*60}")
    print(f"Processing {lang_name} translation...")
    print(f"{'='*60}")
    
    output_path = Path(OUTPUT_DIR) / f"race_{lang_code}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing progress or start fresh
    if output_path.exists():
        print(f"Loading existing progress from {output_path}")
        df = pd.read_csv(output_path)
    else:
        print(f"Loading original CSV from {INPUT_CSV}")
        df = pd.read_csv(INPUT_CSV)
        df['article'] = ""
        df['problems'] = ""
        df['qa'] = ""
    
    # Load original data for reference
    original_df = pd.read_csv(INPUT_CSV)
    
    # Translate each row
    for idx in tqdm(range(len(df)), desc=f"{lang_name}"):
        # Step 1: Translate article if not already done
        if pd.isna(df.at[idx, 'article']) or not df.at[idx, 'article'].strip():
            original_article = original_df.at[idx, 'article']
            df.at[idx, 'article'] = translate_text(original_article, lang_name)
            df.to_csv(output_path, index=False)
        
        # Step 2: Translate problems if not already done
        if pd.isna(df.at[idx, 'problems']) or not df.at[idx, 'problems'].strip():
            original_problems = original_df.at[idx, 'problems']
            df.at[idx, 'problems'] = translate_text(original_problems, lang_name)
            df.to_csv(output_path, index=False)
        
        # Step 3: Smart translate qa by replacing article content
        if pd.isna(df.at[idx, 'qa']) or not df.at[idx, 'qa'].strip():
            original_article = original_df.at[idx, 'article']
            original_qa = original_df.at[idx, 'qa']
            translated_article = df.at[idx, 'article']
            
            # Replace the English article in qa with the translated article
            if pd.notna(original_article) and original_article.strip() in original_qa:
                # Replace the article portion with translated version
                qa_with_translated_article = original_qa.replace(original_article.strip(), translated_article)
                
                # Now translate the remaining parts (questions/answers)
                df.at[idx, 'qa'] = translate_text(qa_with_translated_article, lang_name)
            else:
                # If we can't find the article in qa, translate the whole thing
                df.at[idx, 'qa'] = translate_text(original_qa, lang_name)
            
            df.to_csv(output_path, index=False)
    
    print(f"\n✅ {lang_name} translation completed and saved to {output_path}")

print("\n" + "="*60)
print("All translations completed!")
print("="*60)
