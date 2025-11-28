import pandas as pd
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# all nice, but placeholder issues in later sentences
'''
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


# all nice, but sometimes #### missing
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
            placeholders['PHx'] = match.group(0)
            text = text.replace(match.group(0), ' PHx ')

        # #### <number> placeholder
        match = re.search(r"####\s*\d+", text)
        if match:
            placeholders['PHy'] = match.group(0)
            text = text.replace(match.group(0), ' PHy ')

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
            print(translated_text)
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









'''














import pandas as pd
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Tuple, Dict, Any


# ---------------------------
# Config
# ---------------------------
SENTINEL_LEFT = "꧁"   # rare unicode chars reduce tokenization/transliteration risk
SENTINEL_RIGHT = "꧂"
PH_MATH_PREFIX = "PHMATH"  # will produce e.g. ꧁PHMATH_A꧂
PH_ANS_KEY = f"{SENTINEL_LEFT}PH_ANS{SENTINEL_RIGHT}"
MAX_NEW_TOKENS_DEFAULT = 2048


# ---------------------------
# Helpers
# ---------------------------
def make_ph(id_char: str) -> str:
    return f"{SENTINEL_LEFT}{PH_MATH_PREFIX}_{id_char}{SENTINEL_RIGHT}"


def generate_id_chars(n: int):
    """Yield A..Z, then AA.. if needed (simple base-26 uppercase)."""
    out = []
    i = 0
    while len(out) < n:
        # convert i to letters base-26
        x = i
        s = ""
        while True:
            s = chr(ord('A') + (x % 26)) + s
            x = x // 26 - 1
            if x < 0:
                break
        out.append(s)
        i += 1
    return out


# Patterns to protect as math / important spans
MATH_PATTERNS = [
    r"<<[^<>]*>>",                 # your original <<...>>
    r"\$[^$]+\$",                  # $...$
    r"\\\([^()]*\\\)",             # \(...\)
    r"\\\[[^\]]*\\\]",             # \[...\]
    # Inline TeX fractions, e.g. \frac{1}{2} or simple braces { ... } are harder to protect generically;
    # add capturing for \frac{...}{...} to avoid partial translation.
    r"\\frac\{[^}]*\}\{[^}]*\}",
]


MATH_COMBINED_REGEX = re.compile("|".join(f"({p})" for p in MATH_PATTERNS), flags=re.DOTALL)


# canonical answer regex — captures integers, decimals, signed numbers, simple fractions like 3/4
ANS_REGEX = re.compile(r"####\s*([-+]?\d+(\.\d+)?(?:/\d+)?|\d+)", flags=re.IGNORECASE)


def protect_placeholders(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Replace all math-like matches with unique sentinel placeholders.
    Replace all '#### <answer>' occurrences with a single PH_ANS marker and record canonical original.
    Return (protected_text, placeholders)
    placeholders keys: 'PH_ANS_ORIG' optional, and mapping from placeholder -> original span
    """
    placeholders: Dict[str, Any] = {}

    # first find all math spans with positions to preserve exact text (avoid overlapping replacements)
    matches = list(MATH_COMBINED_REGEX.finditer(text))
    # also include <<...>> with greedy safe alt if none matched (defensive)
    # (already included in patterns)

    # build placeholder ids
    ids = generate_id_chars(len(matches))
    protected = []
    last = 0
    for i, m in enumerate(matches):
        protected.append(text[last:m.start()])
        ph = make_ph(ids[i])
        orig = m.group(0)
        placeholders[ph] = orig
        protected.append(f" {ph} ")
        last = m.end()
    protected.append(text[last:])
    text_with_ph = "".join(protected)

    # canonical answer: find first occurrence only and replace all occurrences with PH_ANS_KEY
    ans_match = ANS_REGEX.search(text_with_ph)
    if ans_match:
        canonical = ans_match.group(0)  # e.g., "#### 48" exactly with formatting
        placeholders['PH_ANS_ORIG'] = canonical
        # replace all variants of #### <something> with PH_ANS_KEY
        text_with_ph = re.sub(r"####\s*[-+]?\d+(\.\d+)?(?:/\d+)?", f" {PH_ANS_KEY} ", text_with_ph, flags=re.IGNORECASE)

    # normalize whitespace
    text_with_ph = re.sub(r"\s{2,}", " ", text_with_ph).strip()
    return text_with_ph, placeholders


def restore_placeholders_and_clean(model_output: str, placeholders: Dict[str, Any], col_name: str = None) -> str:
    """
    Restore placeholders robustly and clean artifacts.
    - Replace sentinel placeholders back to original exact spans.
    - Remove duplicates like <<...=N>> N or similar.
    - Remove stray tokens and hallucinated placeholders.
    - Remove any reintroduced '#### N' and append canonical once at end of 'instruct' column.
    - Remove accidental letters attached to math placeholders.
    """
    text = model_output

    # 1) Restore math placeholders
    for ph, orig in placeholders.items():
        if ph == 'PH_ANS_ORIG':
            continue
        inner = re.escape(ph.strip(SENTINEL_LEFT + SENTINEL_RIGHT))
        pattern = re.compile(re.escape(SENTINEL_LEFT) + r"\s*" + inner + r"\s*" + re.escape(SENTINEL_RIGHT),
                             flags=re.IGNORECASE)
        text = pattern.sub(orig, text)

    # 2) Defensive replacement for common placeholder variants
    for ph in [k for k in placeholders.keys() if k != 'PH_ANS_ORIG']:
        id_inner = ph.strip(SENTINEL_LEFT + SENTINEL_RIGHT)
        variant_regex = re.compile(r"(\[|\(|\{)?\s*__?" + re.escape(id_inner) + r"__?\s*(\]|\)|\})?", flags=re.IGNORECASE)
        text = variant_regex.sub(placeholders[ph], text)
        variant2 = re.compile(r"\b" + r"\s*".join(re.escape(part) for part in id_inner.split("_")) + r"\b", flags=re.IGNORECASE)
        text = variant2.sub(placeholders[ph], text)

    # 3) Remove any remaining "PH-..." hallucinations
    text = re.sub(r"\bPH[_\s\-]?(ANS|MATH)?[A-Z0-9_]*\b", " ", text, flags=re.IGNORECASE)

    # 4) Remove stray words
    stray_words = [r"\bANS\b", r"\bPHANS\b", r"\bPH_H_H_H\b", r"\bPH_MAT?H?_?\d*\b", r"\bMATH_PH[_\d]*\b"]
    for w in stray_words:
        text = re.sub(w, " ", text, flags=re.IGNORECASE)

    # 5) Remove duplicates like <<...=N>> N or <<...=N>>(N) etc.
    def remove_duplicate_after_placeholder(s):
        pattern = re.compile(
            r"(<<[^<>]*?=(\-?\d*\.?\d*)>>)\s*[\)=\(\[\s]*\2[\]\)]*",
            flags=re.IGNORECASE
        )
        return pattern.sub(r"\1", s)
    text = remove_duplicate_after_placeholder(text)

    # 6) Remove accidental letters attached to placeholders
    text = re.sub(r"(<<[^<>]*?>>)[A-Z]{1,3}", r"\1", text)

    # 7) Remove any remaining raw '#### <number>'
    text = re.sub(r"####\s*[-+]?\d+(\.\d+)?(?:/\d+)?", " ", text)

    # 8) Normalize whitespace and punctuation
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"[ \t\n]+([,;:\.\?\!]+)$", r"\1", text)
    text = re.sub(r"[\.]{2,}", ".", text)

    # 9) Append canonical answer for 'instruct' column
    if col_name == 'instruct' and 'PH_ANS_ORIG' in placeholders:
        text = re.sub(r"[ \t\n\r\f\v]+$", "", text)
        text = re.sub(r"[ \t\n\r\f\v\.,;:!?]+$", "", text)
        text = f"{text} {placeholders['PH_ANS_ORIG']}".strip()

    # 10) Final whitespace normalization
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def translate_text(text: str, tokenizer, model, tgt_lang: str, device: str, col_name: str = None,
                   max_new_tokens: int = MAX_NEW_TOKENS_DEFAULT) -> str:
    protected_text, placeholders = protect_placeholders(text)

    encoded = tokenizer(protected_text, return_tensors="pt", truncation=True).to(device)

    # get forced language id robustly (handles different HF tokenizer versions)
    forced_id = None
    if hasattr(tokenizer, "_lang_token_to_id"):
        forced_id = tokenizer._lang_token_to_id.get(tgt_lang, None)
    if forced_id is None:
        try:
            forced_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        except Exception:
            forced_id = None

    gen_kwargs = {
        "forced_bos_token_id": forced_id,
        "max_new_tokens": max_new_tokens,
        "num_beams": 1,
    }
    # drop None entries
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with torch.no_grad():
        outputs = model.generate(**encoded, **gen_kwargs)

    translated_raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned = restore_placeholders_and_clean(translated_raw, placeholders, col_name=col_name)
    return cleaned


# ---------------------------
# Main translation function
# ---------------------------
def translate_gsm8k_unbatched(
    input_csv_path: str,
    output_csv_path: str,
    target_lang: str = "german",
    n_rows: int = None,
    device: str = None,
    model_name: str = "facebook/nllb-200-3.3B",
):
    """
    Robust version of the translator:
      - protects math spans with strong sentinels
      - protects '#### <answer>' and appends canonical once to 'instruct'
      - robust restoration tolerant to small model edits
    """
    lang_code_map = {
        'german': 'deu_Latn',
        'french': 'fra_Latn',
        'spanish': 'spa_Latn'
    }
    if target_lang.lower() not in lang_code_map:
        raise ValueError(f"Unsupported target_lang: choose from {list(lang_code_map.keys())}")
    tgt_lang = lang_code_map[target_lang.lower()]

    df = pd.read_csv(input_csv_path)
    total_rows = len(df) if n_rows is None else min(n_rows, len(df))

    # load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # main loop
    for idx in range(total_rows):
        for col in ['instruct', 'qa']:
            original_text = str(df.at[idx, col])
            print(f"[row {idx}] Translating column '{col}'...")
            try:
                # adjust max_new_tokens heuristically to avoid truncation for longer inputs:
                est_tokens = max(128, min(MAX_NEW_TOKENS_DEFAULT, len(original_text.split()) * 3))
                translated = translate_text(original_text, tokenizer, model, tgt_lang, device, col_name=col,
                                            max_new_tokens=est_tokens)
            except Exception as e:
                print(f"Warning: translation failed for row {idx} col {col}: {e}")
                translated = original_text
            df.at[idx, col] = translated
            print(translated)

    df.to_csv(output_csv_path, index=False)
    print(f"Translation complete. Saved to {output_csv_path}")


# ---------------------------
# Example usage (script)
# ---------------------------
if __name__ == "__main__":
    translate_gsm8k_unbatched(
        input_csv_path="../MathNeuro/data/gsm8k.csv",
        output_csv_path="gsm8k_output_improved.csv",
        target_lang="german",
        n_rows=10
    )

'''






import pandas as pd
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import re





def translate_gsm8k(
    input_csv_path,
    output_csv_path,
    target_lang='german',
    n_rows=None,
    device=None
    ):


    # Language map
    lang_code_map = {
        'german': 'deu_Latn',
        'french': 'fra_Latn',
        'spanish': 'spa_Latn'
    }
    if target_lang.lower() not in lang_code_map:
        raise ValueError(f"Unsupported target_lang: choose from {list(lang_code_map.keys())}")
    tgt_lang = lang_code_map[target_lang.lower()]

    # Load CSV
    df = pd.read_csv(input_csv_path)
    total_rows = len(df) if n_rows is None else min(n_rows, len(df))

    # Load model
    model_name = "facebook/nllb-200-3.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # -------------------------
    # Helper functions
    # -------------------------

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
        translate_func: your existing translate_text function
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

        # NLLB forced target language
        try:
            forced_id = tokenizer._lang_token_to_id[tgt_lang]
        except:
            forced_id = tokenizer.convert_tokens_to_ids(tgt_lang)

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                forced_bos_token_id=forced_id,
                max_new_tokens=2048,
                num_beams=10
            )

        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated = restore_placeholders(translated, placeholders)
        return translated

    # -------------------------
    # Translation loop
    # -------------------------
    for idx in range(total_rows):
        for col in ['instruct', 'qa']:
            original_text = str(df.at[idx, col])
            print(f"Translating row {idx}, column '{col}'...")
            #df.at[idx, col] = translate_text(original_text)
            df.at[idx, col] = translate_long_text(original_text, translate_text)

            print(df.at[idx, col])

    # Save result
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Translation complete. Saved to {output_csv_path}")

if __name__ == "__main__":
    translate_gsm8k(
        input_csv_path="../MathNeuro/data/gsm8k.csv",
        output_csv_path="gsm8k_output_improved.csv",
        target_lang="german",
        n_rows=None
    )