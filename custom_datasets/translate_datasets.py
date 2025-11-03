import pandas as pd
import re
from transformers import MarianMTModel, MarianTokenizer
import torch



def translate_gsm8k(input_csv_path, output_csv_path, target_lang='german', n_rows=None):
    """
    Translate a GSM8K CSV from English to a target language, preserving math placeholders and answer markers.

    Args:
        input_csv_path (str): Path to the input CSV (GSM8K format).
        output_csv_path (str): Path where the translated CSV will be saved.
        target_lang (str): Target language ('german', 'french', 'spanish').
        n_rows (int, optional): Number of rows to translate for debugging. Translate all if None.
    """
    # Map language to MarianMT model names
    lang_model_map = {
        'german': 'Helsinki-NLP/opus-mt-en-de',
        'french': 'Helsinki-NLP/opus-mt-en-fr',
        'spanish': 'Helsinki-NLP/opus-mt-en-es'
    }

    if target_lang.lower() not in lang_model_map:
        raise ValueError(f"Unsupported target language: {target_lang}. Choose from {list(lang_model_map.keys())}")

    model_name = lang_model_map[target_lang.lower()]

    # Load CSV
    df = pd.read_csv(input_csv_path)

    # Load translation model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    def protect_math_and_markers(text):
        """Protect <<…>> expressions and #### markers."""
        placeholders = {}

        # Protect <<…>> placeholders
        def repl_math(m):
            key = f"__PH_{len(placeholders)}__"
            placeholders[key] = m.group(0)
            return key

        text_protected = re.sub(r"<<.*?>>", repl_math, text)

        # Protect #### markers
        def repl_hash(m):
            key = f"__PH_{len(placeholders)}__"
            placeholders[key] = m.group(0)
            return key

        text_protected = re.sub(r"####", repl_hash, text_protected)

        return text_protected, placeholders

    def restore_placeholders(text, placeholders):
        """Restore placeholders after translation."""
        for key, val in placeholders.items():
            text = text.replace(key, val)
        return text

    def translate_text(text):
        text_protected, placeholders = protect_math_and_markers(text)
        batch = tokenizer([text_protected], return_tensors="pt", padding=True)
        translated_tokens = model.generate(**batch)
        translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens][0]
        return restore_placeholders(translated_text, placeholders)

    # Determine rows to translate
    total_rows = len(df) if n_rows is None else min(n_rows, len(df))

    # Translate
    for idx in range(total_rows):
        for col in ['instruct', 'qa']:
            print(f"Translating row {idx}, column {col} to {target_lang}...")
            df.at[idx, col] = translate_text(df.at[idx, col])

    # Save the translated CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Translation complete. Saved to {output_csv_path}")



'''
# Translate first 100 rows to German
translate_gsm8k(
    input_csv_path="../MathNeuro/data/gsm8k.csv",
    output_csv_path="translated_german.csv",
    target_lang="german",
    n_rows=4
)

# Translate the full dataset to French
translate_gsm8k(
    input_csv_path="../MathNeuro/data/gsm8k.csv",
    output_csv_path="translated_french.csv",
    target_lang="french",
    n_rows=4
)

# Translate the full dataset to Spanish
translate_gsm8k(
    input_csv_path="../MathNeuro/data/gsm8k.csv",
    output_csv_path="translated_spanish.csv",
    target_lang="spanish",
    n_rows=4
)
'''



def translate_race_dataset(
        input_csv_path,
        output_csv_path,
        target_lang='german',
        columns_to_translate=('article', 'problems', 'qa'),
        n_rows=None,
        batch_size=8
):
    """
    Translate RACE dataset to a target language (German, French, or Spanish) using MarianMT.

    Args:
        input_csv_path (str): Path to input CSV.
        output_csv_path (str): Path to save translated CSV.
        target_lang (str): Target language ('german', 'french', 'spanish').
        columns_to_translate (tuple): Columns to translate.
        n_rows (int, optional): Limit number of rows (for testing).
        batch_size (int): Number of samples to translate per batch.
    """

    # --- Load dataset ---
    df = pd.read_csv(input_csv_path)
    if n_rows is not None:
        df = df.head(n_rows)

    # --- Define language model map ---
    model_map = {
        'german': 'Helsinki-NLP/opus-mt-en-de',
        'french': 'Helsinki-NLP/opus-mt-en-fr',
        'spanish': 'Helsinki-NLP/opus-mt-en-es'
    }
    if target_lang.lower() not in model_map:
        raise ValueError(f"Unsupported target language: {target_lang}")

    model_name = model_map[target_lang.lower()]

    # --- Load model + tokenizer ---
    print(f"Loading model {model_name} ...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # --- Move to GPU if available ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --- Translation helper ---
    def translate_batch(texts):
        """Translate a list of texts as a batch."""
        batch = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            translated = model.generate(**batch)
        return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    # --- Translate in batches ---
    total_rows = len(df)
    for col in columns_to_translate:
        print(f"\nTranslating column '{col}' to {target_lang}...")
        translated_texts = []

        for i in range(0, total_rows, batch_size):
            batch_texts = df[col].iloc[i:i + batch_size].fillna("").tolist()
            translations = translate_batch(batch_texts)
            translated_texts.extend(translations)

            print(f"  → Translated {i + len(batch_texts)} / {total_rows} rows", end="\r")

        df[col] = translated_texts

    # --- Save output ---
    df.to_csv(output_csv_path, index=False)
    print(f"\n✅ Translation complete! Saved to: {output_csv_path}")



translate_race_dataset(
    input_csv_path="../MathNeuro/data/race.csv",
    output_csv_path="race_german.csv",
    target_lang="german",
    n_rows=10,   # set None to translate all
    batch_size=4  # increase if you have a GPU with more memory
)
