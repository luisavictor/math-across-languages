'''
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
    n_rows=10    # subset for testing
)
'''








'''
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
    Strictly preserves exactly one <<...>> and one #### <number> per row per column.
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
    def protect_placeholders(text):
        placeholders = {}

        # <<...>> placeholder
        match = re.search(r"<<.*?>>", text)
        if match:
            placeholders['PH_MATH'] = match.group(0)
            text = text.replace(match.group(0), ' PH_MATH ')

        # #### <number> placeholder
        match = re.search(r"####\s*\d+", text)
        if match:
            placeholders['PH_ANS'] = match.group(0)
            text = text.replace(match.group(0), ' PH_ANS ')

        return text.strip(), placeholders

    def restore_placeholders(text, placeholders):
        for key, val in placeholders.items():
            text = text.replace(key, val)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

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
                num_beams=1
            )

        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated = restore_placeholders(translated, placeholders)
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
    n_rows=10
)
'''





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
    Translation pipeline that:
      - protects <<...>> math placeholders exactly and restores them unchanged
      - replaces all '#### <number>' occurrences with a single canonical marker before translation,
        then appends the canonical '#### <number>' exactly once at the end of the 'instruct' text
      - prevents duplicates like <<32+16=48>>48
      - strips stray placeholder artifacts (ANS, PHANS, malformed PH tokens)
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
    def protect_placeholders(text):
        """
        Replace all <<...>> with unique keys, and replace ALL '#### <number>' occurrences
        with a single marker '__PH_ANS__'. Record canonical answer (first ####).
        Return (protected_text, placeholders_dict).
        placeholders_dict contains:
           - math keys mapping to original <<...>> strings
           - 'PH_ANS_ORIG' if an answer existed
        """
        placeholders = {}

        # protect math placeholders (all occurrences)
        math_matches = re.findall(r"<<.*?>>", text)
        for i, m in enumerate(math_matches):
            key = f"__PH_MATH_{i}__"
            placeholders[key] = m
            text = text.replace(m, f" {key} ")

        # canonical answer: capture first #### N (if any) and replace ALL with marker
        ans_match = re.search(r"####\s*\d+", text)
        if ans_match:
            placeholders['PH_ANS_ORIG'] = ans_match.group(0)
            # replace all occurrences of #### <number> with marker to avoid duplications by model
            text = re.sub(r"####\s*\d+", " __PH_ANS__ ", text)

        return text.strip(), placeholders

    def restore_placeholders_and_clean(model_output, placeholders, col_name=None):
        """
        1) Replace math keys with original <<...>> exactly.
        2) Remove duplicates like <<...=N>>N (where N equals the number after =).
        3) Remove stray tokens such as ANS, PHANS, stray __PH_* tokens.
        4) Remove any remaining '#### N' (model might reintroduce) and append the canonical once to end of 'instruct'.
        """
        text = model_output

        # Replace math markers back to original (do this before other cleanup)
        for key, orig in placeholders.items():
            if key.startswith("__PH_MATH_"):
                # replace any variant forms (e.g., model slightly altered key) defensively:
                # replace exact key first, then try variations
                text = text.replace(key, orig)

        # Defensive: remove any model-inserted malformed PH keys (very conservative)
        text = re.sub(r"__PH_MATH_\d+__", lambda m: placeholders.get(m.group(0), ""), text)
        text = re.sub(r"__PH_ANS__\b", " ", text)

        # Remove typical stray words the model sometimes emits
        text = re.sub(r"\bANS\b", " ", text)
        text = re.sub(r"\bPHANS\b", " ", text)
        text = re.sub(r"\bPH_H_H_H\b", " ", text)
        text = re.sub(r"\bPH_MAT?H?_?\d*\b", " ", text)
        text = re.sub(r"\bMATH_PH[_\d]*\b", " ", text)

        # Remove duplicates like <<...=N>>N  -> keep <<...=N>>
        # We identify <<...=N>> then look for immediate following same number (possibly with space/punct)
        def remove_dup_after_placeholder(s):
            # replace occurrences of <<...=N>> <optional punctuation/spaces> N  -> <<...=N>>
            return re.sub(r"(<<[^<>]*?=(\-?\d+\.?\d*)>>)([\s\.,]*)\2\b", r"\1", s)
        text = remove_dup_after_placeholder(text)

        # Remove any remaining raw '#### N' occurrences (we will append canonical one later if present)
        text = re.sub(r"####\s*\d+", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s{2,}", " ", text).strip()

        # Append canonical answer for instruct only (if original had one)
        if col_name == 'instruct' and 'PH_ANS_ORIG' in placeholders:
            # ensure no trailing punctuation conflicts
            text = text.rstrip(' .,:;')
            text = f"{text} {placeholders['PH_ANS_ORIG']}"

        return text.strip()

    def translate_text(text, col_name=None):
        protected_text, placeholders = protect_placeholders(text)

        # encode & translate
        encoded = tokenizer(protected_text, return_tensors="pt", truncation=True).to(device)
        try:
            forced_id = tokenizer._lang_token_to_id[tgt_lang]
        except Exception:
            forced_id = tokenizer.convert_tokens_to_ids(tgt_lang)

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                forced_bos_token_id=forced_id,
                max_new_tokens=2048,
                num_beams=1
            )

        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        cleaned = restore_placeholders_and_clean(translated, placeholders, col_name=col_name)
        return cleaned

    # ---------------------------
    # MAIN LOOP — UNBATCHED
    # ---------------------------
    for idx in range(total_rows):
        for col in ['instruct', 'qa']:
            original_text = str(df.at[idx, col])
            print(f"Translating row {idx}, column '{col}'...")
            try:
                translated_text = translate_text(original_text, col_name=col)
            except Exception as e:
                print(f"Warning: translation failed for row {idx} col {col}: {e}")
                translated_text = original_text

            df.at[idx, col] = translated_text

    # ---------------------------
    # SAVE
    # ---------------------------
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Translation complete. Saved to {output_csv_path}")


# Example usage (run as script)
if __name__ == "__main__":
    translate_gsm8k_unbatched(
        input_csv_path="../MathNeuro/data/gsm8k.csv",   # change to your path
        output_csv_path="gsm8k_output.csv",
        target_lang="german",
        n_rows=10
    )

