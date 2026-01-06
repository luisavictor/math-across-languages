import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re


def translate_gsm8k(
    input_csv_path,
    output_csv_path,
    text_columns,
    target_lang='german',
    n_rows=None,
    device=None
    ):

    # Language map
    lang_code_map = {
        'german': 'deu_Latn',
        'french': 'fra_Latn',
        'spanish': 'spa_Latn',
        'hindi': 'hin_Deva',
    }
    if target_lang.lower() not in lang_code_map:
        raise ValueError(f"Unsupported target_lang: choose from {list(lang_code_map.keys())}")
    tgt_lang = lang_code_map[target_lang.lower()]

    # Load CSV
    df = pd.read_csv(input_csv_path)
    total_rows = len(df) if n_rows is None else min(n_rows, len(df))

    # Load model
    model_name = "facebook/nllb-200-3.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Helper functions
    def split_text_into_chunks(text):
        """
        Split a long sentence into chunks at punctuation or logical connectors.
        Keeps punctuation attached to the preceding clause.
        """
        # First, split at sentence punctuation (., ?, !)
        sentence_chunks = re.split(r'(?<=[.?!])\s+', text)

        # Further split long clauses by logical connectors
        final_chunks = []
        for chunk in sentence_chunks:
            # Split at common connectors only if the chunk is still long
            if len(chunk) > 150:  # threshold for “long chunk”
                sub_chunks = re.split(r'(?<=,)\s+(then|because|so|and)\s+', chunk, flags=re.IGNORECASE)
                final_chunks.extend([c.strip() for c in sub_chunks if c.strip()])
            else:
                final_chunks.append(chunk.strip())

        return final_chunks

    def translate_long_text(text, translate_func):
        """
        Split text into chunks, translate each separately, then join back.
        """
        chunks = split_text_into_chunks(text)
        translated_chunks = []
        for chunk in chunks:
            translated_chunks.append(translate_func(chunk))
        # Join translated chunks with a space
        return ' '.join(translated_chunks)
    def protect_placeholders(text):
        """
        Replace <<...>> and #### <number> with safe tokens before translation.
        """
        placeholders = {}
        # Protect <<...>> placeholders
        for i, match in enumerate(re.findall(r"<<.*?>>", text)):
            key = f"PHX{i}"
            placeholders[key] = match
            text = text.replace(match, f" {key} ")

        # Protect #### <number> placeholders
        for i, match in enumerate(re.findall(r"####\s*\d+", text)):
            key = f"PHY{i}"
            placeholders[key] = match
            text = text.replace(match, f" {key} ")

        return text.strip(), placeholders

    def restore_placeholders(text, placeholders):
        """
        Restore placeholders exactly to their original form.
        """
        for key, val in placeholders.items():
            text = text.replace(key, val)
        # Clean up extra spaces
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def translate_text(text):
        protected_text, placeholders = protect_placeholders(text)
        encoded = tokenizer(protected_text, return_tensors="pt", truncation=True).to(device)

        try:
            forced_id = tokenizer._lang_token_to_id[tgt_lang]
        except:
            forced_id = tokenizer.convert_tokens_to_ids(tgt_lang)

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                forced_bos_token_id=forced_id,
                max_new_tokens=2048,
                num_beams=8
            )

        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated = restore_placeholders(translated, placeholders)
        return translated

    # Translation loop
    for idx in range(total_rows):
        for col in text_columns:
            print(f"Translating row {idx}, column '{col}'...")
            df.at[idx, col] = translate_long_text(str(df.at[idx, col]), translate_text)
        df.to_csv(output_csv_path, index=False)

    print(f"Translation complete. Saved to {output_csv_path}")

    # Save result
    df.to_csv(output_csv_path, index=False)





translate_gsm8k(
    input_csv_path="../../MathNeuro/data/gsm8k_test.csv",
    output_csv_path="../GSM8k_Hindu/gsm8k_hi_test.csv",
    text_columns=["question", "answer"],
    target_lang="hindi",
    n_rows=100 # allows to debug/test translations on the first n_rows only, ignores limit for "None"
)



'''
translate_gsm8k(
    input_csv_path="../../MathNeuro/data/gsm8k.csv",
    output_csv_path="gsm8k_de_train.csv",
    text_columns=["instruct", "qa"],
    target_lang="german",
    n_rows=10
)

# test dataset translation from original huggingface test set (only #### XY option)
translate_gsm8k(
    input_csv_path="../../MathNeuro/data/gsm8k_test.csv",
    output_csv_path="gsm8k_de_test.csv",
    text_columns=["question", "answer"],
    target_lang="german",
    n_rows=None # allows to debug/test translations on the first n_rows only, ignores limit for "None"
)



translate_gsm8k(
    input_csv_path="../../MathNeuro/data/gsm8k.csv",
    output_csv_path="../GSM8k_Hindu/gsm8k_hi_train.csv",
    text_columns=["instruct", "qa"],
    target_lang="hindi",
    n_rows=100
)




'''