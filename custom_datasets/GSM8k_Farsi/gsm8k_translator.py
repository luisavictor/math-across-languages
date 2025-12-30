import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import re


def translate_gsm8k(
    input_csv_path,
    output_csv_path,
    text_columns,
    n_rows=None,
    device=None
    ):

    # Load CSV
    df = pd.read_csv(input_csv_path)
    total_rows = len(df) if n_rows is None else min(n_rows, len(df))

    # Load model
    model_name = "Sheikhaei/llama-3.2-1b-en-fa-translator"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, 
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
            if len(chunk) > 1024:  # threshold for “long chunk”
                sub_chunks = re.split(r'(?<=,)\s+(then|because|so|and)\s+', chunk, flags=re.IGNORECASE)
                final_chunks.extend([c.strip() for c in sub_chunks if c.strip()])
            else:
                final_chunks.append(chunk.strip())

        return final_chunks
    
    def replace_digits_with_persian(text):
        """
        Replace Western Arabic numerals with Persian numerals.
        """
        western_to_persian_digits = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')
        return text.translate(western_to_persian_digits)

    def replace_q_a_with_persian(text):
        """
        Replace 'Q:' and 'A:' with Persian equivalents.
        """
        text = re.sub(r'\bQ:\b', 'س:', text)
        text = re.sub(r'\bA:\b', 'ج:', text)
        return text

    def translate_long_text(text, translate_func):
        """
        Split text into chunks, translate each separately, then join back.
        """
        chunks = split_text_into_chunks(text)
        translated_chunks = []
        for chunk in chunks:
            translation = translate_func(chunk)
            translation = replace_digits_with_persian(translation)
            translation = replace_q_a_with_persian(translation)
            translated_chunks.append(translation)
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

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=2048,    
                do_sample=False,
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




# training datset translation (English version from MathNeurosurgery repo, but same sentences as from Huggingface with different column names and additional "The answer is XY".
translate_gsm8k(
    input_csv_path="MathNeuro/data/gsm8k.csv",
    output_csv_path="custom_datasets/GSM8k_Farsi/gsm8k_fa_train.csv",
    text_columns=["instruct", "qa"],
    n_rows=None
)

# test dataset translation from original huggingface test set (only #### XY option)
translate_gsm8k(
    input_csv_path="MathNeuro/data/gsm8k_test.csv",
    output_csv_path="custom_datasets/GSM8k_Farsi/gsm8k_fa_test.csv",
    text_columns=["question", "answer"],
    n_rows=None # allows to debug/test translations on the first n_rows only, ignores limit for "None"
)
