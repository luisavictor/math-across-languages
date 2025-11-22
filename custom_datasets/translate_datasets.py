import pandas as pd
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def translate_gsm8k_unbatched(
    input_csv_path,
    output_csv_path,
    target_lang='german',
    n_rows=None,
    device=None
):
    """
    Sentence-by-sentence translation of GSM8K-like CSV to a target language using NLLB-200.
    Preserves <<...>> and #### <number> markers robustly.
    """

    # ---------------------------
    # Language Map
    # ---------------------------
    lang_code_map = {
        'german': 'deu_Latn',
        'french': 'fra_Latn',
        'spanish': 'spa_Latn'
    }
    if target_lang.lower() not in lang_code_map:
        raise ValueError(f"Unsupported target_lang: choose from {list(lang_code_map.keys())}")

    tgt_lang = lang_code_map[target_lang.lower()]

    # ---------------------------
    # Load Data
    # ---------------------------
    df = pd.read_csv(input_csv_path)
    total_rows = len(df) if n_rows is None else min(n_rows, len(df))

    # ---------------------------
    # Load Model
    # ---------------------------
    model_name = "facebook/nllb-200-3.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # ---------------------------
    # Helper functions
    # ---------------------------
    def protect_math_and_markers(text):
        placeholders = {}
        next_id = 0

        def repl_marker(m):
            nonlocal next_id
            key = f"<PH{next_id}>"
            placeholders[key] = m.group(0)
            next_id += 1
            return f" {key} "

        text = re.sub(r"####\s*\d+", repl_marker, text)
        text = re.sub(r"####", repl_marker, text)
        text = re.sub(r"<<.*?>>", repl_marker, text)
        return text.strip(), placeholders

    def restore_placeholders(translated_text, placeholders):
        text = translated_text
        for key, val in placeholders.items():
            text = text.replace(key, val)

        missing = [val for key, val in placeholders.items()
                   if re.match(r"^####\s*\d+$", val) or val.strip() == "####" and val not in text]

        text = re.sub(r"(####\s*\d+)(\s*####\s*\d+)+", r"\1", text)
        text = re.sub(r'\s*<<\s*', '<<', text)
        text = re.sub(r'\s*>>\s*', '>>', text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text, missing

    def reinsert_missing_markers(text, missing_markers):
        for marker in missing_markers:
            if marker in text:
                continue
            m = re.search(r'([.!?])\s*$', text)
            if m:
                text = re.sub(r'([.!?])\s*$', r'\1 ' + marker, text)
            else:
                text = text.rstrip() + " " + marker
        text = re.sub(r"(####\s*\d+)(\s*####\s*\d+)+", r"\1", text)
        return re.sub(r"\s{2,}", " ", text).strip()

    def translate_text(text):
        protected_text, placeholders = protect_math_and_markers(text)
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
                num_beams=1  # can reduce beams for speed
            )

        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated, missing = restore_placeholders(translated, placeholders)
        if missing:
            translated = reinsert_missing_markers(translated, missing)
        return translated

    # ---------------------------
    # MAIN LOOP — UNBATCHED
    # ---------------------------
    for idx in range(total_rows):
        for col in ['instruct', 'qa']:
            original_text = str(df.at[idx, col])
            print(f"Translating row {idx}, column '{col}'...")
            translated_text = translate_text(original_text)
            df.at[idx, col] = translated_text

    # ---------------------------
    # SAVE
    # ---------------------------
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Translation complete. Saved to {output_csv_path}")


# Example usage
translate_gsm8k_unbatched(
    input_csv_path="../MathNeuro/data/gsm8k.csv",
    output_csv_path="gsm8k_german.csv",
    target_lang="german",
    n_rows=None # subset for testing
)
