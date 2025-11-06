from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import re

# ---- Configuration ----
TARGET_LANGUAGES = {
    "de": "German",
    #"fr": "French",
    #"es": "Spanish",
    #"it": "Italian",
    #"nl": "Dutch",
    #"pt": "Portuguese"
}

MODEL_NAME = "utter-project/EuroLLM-1.7B-Instruct"
INPUT_CSV = "MathNeuro/data/race.csv"  # Update this to your actual input file path if needed
OUTPUT_DIR = "C:/Users/timse/Documents/Lab_test/LabTest"

# ---- Step 1: Load quantized model ----
print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, model_max_length=4096)
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
    max_new_tokens=2048,
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.2
)

# ---- Step 2: Translation function ----
def translate_text(text, target_language):
    """Translate text to target language"""
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return ""
    
    # Use a clearer, more direct prompt format
    prompt = f"Translate this English text to {target_language}:\n\n{text.strip()}\n\n{target_language} translation:"
    
    try:
        # Limit max tokens based on input length
        max_tokens = min(len(text) * 2, 2048)
        out = translator(prompt, return_full_text=False, max_new_tokens=max_tokens)[0]["generated_text"]
        
        # Extract only the first line/paragraph of translation
        translation = out.strip()
        
        # Stop at double newlines or when repetition starts
        if '\n\n' in translation:
            translation = translation.split('\n\n')[0]
        
        # Remove any repetition of the original English text
        if text.strip() in translation:
            parts = translation.split(text.strip())
            translation = parts[0] if parts[0].strip() else (parts[1] if len(parts) > 1 else translation)
        
        return translation.strip()
    except Exception as e:
        print(f"Error during translation: {e}")
        return ""

def translate_qa_field(qa_text, translated_article, target_language):
    """Parse QA field, reuse translated article, and translate only Q&A parts"""
    if pd.isna(qa_text) or not isinstance(qa_text, str) or not qa_text.strip():
        return ""
    
    # Pattern: "Read this passage... \n\n [ARTICLE] \n\n Question: ... \n\n Answer choices: ... \n\n Answer: ..."
    
    # Find where "Question:" starts (after the article text)
    question_start = qa_text.find("\n\nQuestion:")
    if question_start == -1:
        # Fallback: just translate the whole thing
        return translate_text(qa_text, target_language)
    
    # Extract just the Q&A part (everything from "Question:" onwards)
    qa_part = qa_text[question_start:].strip()
    
    # Extract question
    question_match = re.search(r'Question:\s*(.+?)(?=\n\nAnswer choices:)', qa_part, re.DOTALL)
    if not question_match:
        return translate_text(qa_text, target_language)
    
    question = question_match.group(1).strip()
    
    # Extract answer choices
    choices_match = re.search(r'Answer choices:\s*(\[.+?\])(?=\n\nAnswer:)', qa_part, re.DOTALL)
    choices = choices_match.group(1).strip() if choices_match else ""
    
    # Extract answer
    answer_match = re.search(r'Answer:\s*(.+?)$', qa_part, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""
    
    # Translate only the intro, question and answer
    intro = "Read this passage and answer the multiple choice question below it."
    intro_translated = translate_text(intro, target_language)
    question_translated = translate_text(question, target_language)
    answer_translated = translate_text(answer, target_language)
    
    # Reconstruct the QA field with the ALREADY TRANSLATED article
    result = f"{intro_translated}\n\n{translated_article}\n\nQuestion: {question_translated}\n\n"
    if choices:
        result += f"Answer choices: {choices}\n\n"
    result += f"Answer: {answer_translated}"
    
    return result

# ---- Step 3: Process each language ----
for lang_code, lang_name in TARGET_LANGUAGES.items():
    print(f"\n{'='*60}")
    print(f"Processing {lang_name} translation...")
    print(f"{'='*60}")
    
    output_path = Path(OUTPUT_DIR) / f"race_{lang_code}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load original data
    print(f"Loading original CSV from {INPUT_CSV}")
    original_df = pd.read_csv(INPUT_CSV)
    
    # Create fresh dataframe for translation
    df = pd.DataFrame()
    df['article'] = ""
    df['problems'] = ""
    df['qa'] = ""
    
    # Copy non-translation columns from original
    for col in original_df.columns:
        if col not in ['article', 'problems', 'qa']:
            df[col] = original_df[col]
    
    # Ensure we have the same number of rows
    df = df.reindex(range(len(original_df)))
    
    # Translate each row
    for idx in tqdm(range(len(df)), desc=f"{lang_name}"):
        row_changed = False
        translated_article = ""
        
        # Step 1: Translate article
        if idx < len(original_df):
            original_article = original_df.at[idx, 'article']
            translated_article = translate_text(original_article, lang_name)
            df.at[idx, 'article'] = translated_article
            row_changed = True
        
        # Step 2: Translate problems
        if idx < len(original_df):
            original_problems = original_df.at[idx, 'problems']
            translated = translate_text(original_problems, lang_name)
            df.at[idx, 'problems'] = translated
            row_changed = True
        
        # Step 3: Translate qa (special handling - reuse translated article)
        if idx < len(original_df):
            original_qa = original_df.at[idx, 'qa']
            translated = translate_qa_field(original_qa, translated_article, lang_name)
            df.at[idx, 'qa'] = translated
            row_changed = True
        
        # Save progress every row
        if row_changed:
            df.to_csv(output_path, index=False)
    
    # Final save
    df.to_csv(output_path, index=False)
    print(f"\n✅ {lang_name} translation completed and saved to {output_path}")

print("\n" + "="*60)
print("All translations completed!")
print("="*60)