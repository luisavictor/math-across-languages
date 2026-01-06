import argparse
import ast
import json
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig


QA_LABELS = {
    "deu_Latn": ("Frage", "Antwortoptionen", "Antwort"),
    "hin_Deva": ("प्रश्न", "उत्तर विकल्प", "उत्तर"),
}

EXCLUDED_SUBJECTS = {
    "abstract_algebra",
    "astronomy",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "conceptual_physics",
    "econometrics",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "professional_accounting",
}

DEFAULT_MODEL_NAME = "facebook/nllb-200-3.3B"
#TARGET_LANG_CODE = "deu_Latn"
TARGET_LANG_CODE = "hin_Deva"



def strip_outer_quotes(text: str) -> str:
    if not isinstance(text, str):
        return text
    cleaned = text.replace('""', '"').replace("''", "'").strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1]
    return cleaned.strip()


def format_choices_mmlu(choices: Sequence[str]) -> str:
    """
    Return a string in the same style as the original MMLU choices:
    ["a", "b"] -> "['a', 'b']" with single quotes and no doubled quotes.
    """
    normalized = []
    for choice in choices:
        c = strip_outer_quotes(choice)
        # avoid doubled quotes in the serialized string
        c = c.replace('"', "'")
        normalized.append(c)
    return "[" + ", ".join(f"'{c}'" for c in normalized) + "]"


def parse_choices(raw_choices: str) -> List[str]:
    try:
        parsed = ast.literal_eval(raw_choices)
        if isinstance(parsed, dict):
            return [str(value) for value in parsed.values()]
        return [str(value) for value in parsed]
    except Exception:
        cleaned = raw_choices.strip("[]")
        return [part.strip(" '\"") for part in cleaned.split(",") if part.strip()]


def split_text_into_chunks(text: str, hard_limit: int = 180) -> List[str]:
    # Short questions and answers translate fine as a single chunk, but keep a
    # splitter to guard against rare long inputs.
    if len(text) <= hard_limit:
        return [text]
    chunks: List[str] = []
    for sentence in text.split(". "):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > hard_limit:
            chunks.extend(part.strip() for part in sentence.split(",") if part.strip())
        else:
            chunks.append(sentence)
    return chunks



'''
def build_qa_text(question: str, choices: Sequence[str], answer_idx: int) -> str:
    # Match original MMLU style: Python list string with single quotes
    formatted_choices = format_choices_mmlu(choices)
    try:
        answer_text = choices[int(answer_idx)]
    except Exception:
        answer_text = ""
    return (
        f"Frage: {question}\n\n"
        f"Antwortoptionen: {formatted_choices}\n\n"
        f"Antwort: {answer_text}"
    )
    
'''

def build_qa_text(
    question: str,
    choices: Sequence[str],
    answer_idx: int,
    target_lang_code: str = "hin_Deva",
) -> str:
    formatted_choices = format_choices_mmlu(choices)
    try:
        answer_text = choices[int(answer_idx)]
    except Exception:
        answer_text = ""
    q_label, c_label, a_label = QA_LABELS.get(
        target_lang_code, ("Question", "Choices", "Answer")
    )
    return (
        f"{q_label}: {question}\n\n"
        f"{c_label}: {formatted_choices}\n\n"
        f"{a_label}: {answer_text}"
    )




def load_model(
    model_name: str,
    target_lang: str,
    hf_token: Optional[str],
    load_in_8bit: bool,
    device_preference: str,
):
    tokenizer_kwargs = {"use_fast": True, "src_lang": "eng_Latn"}
    if device_preference not in {"auto", "cuda", "cpu"}:
        raise ValueError("device_preference must be one of: auto, cuda, cpu")

    if device_preference == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but no GPU detected; falling back to CPU.")
        device_preference = "cpu"

    model_kwargs = {"device_map": device_preference}

    if hf_token:
        tokenizer_kwargs["token"] = hf_token
        model_kwargs["token"] = hf_token

    if load_in_8bit and device_preference != "cpu":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        model_kwargs["torch_dtype"] = (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    try:
        forced_id = tokenizer._lang_token_to_id[target_lang]
    except Exception:
        forced_id = tokenizer.convert_tokens_to_ids(target_lang)

    return tokenizer, model, forced_id


def translate_chunk(
    text: str,
    tokenizer,
    model,
    forced_bos_token_id: int,
    max_input: int = 512,
    max_new_tokens: int = 256,
) -> str:
    if not text:
        return ""
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input,
    )
    target_device = getattr(model, "device", None) or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    encoded = encoded.to(target_device)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=max_new_tokens,
            num_beams=8,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def translate_text(
    text: str,
    tokenizer,
    model,
    forced_bos_token_id: int,
) -> str:
    chunks = split_text_into_chunks(text)
    translated = [
        translate_chunk(chunk, tokenizer, model, forced_bos_token_id) for chunk in chunks
    ]
    return " ".join(part for part in translated if part).strip()


def translate_rows(
    df: pd.DataFrame,
    tokenizer,
    model,
    forced_bos_token_id: int,
    save_every: int,
    output_path: Path,
) -> pd.DataFrame:
    translated_rows = []
    progress = tqdm(df.iterrows(), total=len(df), desc="Translating rows", unit="row")
    for idx, row in progress:
        choices = parse_choices(str(row["choices"]))
        translated_choices = [
            strip_outer_quotes(
                translate_text(choice, tokenizer, model, forced_bos_token_id)
            )
            for choice in choices
        ]
        question_de = strip_outer_quotes(
            translate_text(str(row["question"]), tokenizer, model, forced_bos_token_id)
        )
        qa_de = build_qa_text(question_de, translated_choices, row["answer"])

        translated_rows.append(
            {
                "question": question_de,
                "subject": row["subject"],
                # Store choices as Python list string (single-quoted) like original MMLU
                "choices": format_choices_mmlu(translated_choices),
                "answer": row["answer"],
                "qa": qa_de,
            }
        )

        if save_every and (idx + 1) % save_every == 0:
            pd.DataFrame(translated_rows).to_csv(output_path, index=False, encoding="utf-8")
            print(f"Saved progress after {idx + 1} rows to {output_path}")

    return pd.DataFrame(translated_rows)


def load_and_filter(input_csv: Path, max_rows: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if max_rows:
        df = df.head(max_rows)

    df["subject"] = df["subject"].str.strip()
    before = len(df)
    df = df[~df["subject"].isin(EXCLUDED_SUBJECTS)].reset_index(drop=True)
    removed = before - len(df)
    print(f"Dropped {removed} rows from excluded subjects; {len(df)} rows remain.")
    return df


def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_input = Path(__file__).resolve().parent / "../../MathNeuro/data/mmlu.csv"
    default_input = default_input.resolve()

    #default_output = Path(__file__).resolve().parent / "mmlu_de_test.csv"
    default_output = (Path(__file__).resolve().parent / "../MMLU_Hindu/mmlu_hi_test.csv").resolve()
    default_output.parent.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Translate MMLU to German with NLLB-200.")
    parser.add_argument("--input_csv", type=Path, default=default_input)
    parser.add_argument("--output_csv", type=Path, default=default_output)
    parser.add_argument("--max_rows", type=int, default=1000, help="Limit rows for a quick test run.")
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="Save progress every N rows to avoid losing long runs.",
    )
    parser.add_argument("--hf_token", type=str, default=None, help="Optional HF token for gated models.")
    parser.add_argument(
        "--no_8bit",
        action="store_true",
        help="Disable 8-bit loading if you prefer full precision.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device map to use: cuda, cpu, or auto.",
    )
    args = parser.parse_args()

    df = load_and_filter(args.input_csv, args.max_rows)
    tokenizer, model, forced_id = load_model(
        DEFAULT_MODEL_NAME,
        TARGET_LANG_CODE,
        args.hf_token,
        load_in_8bit=not args.no_8bit,
        device_preference=args.device,
    )

    translated_df = translate_rows(
        df,
        tokenizer,
        model,
        forced_id,
        save_every=args.save_every,
        output_path=args.output_csv,
    )
    translated_df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"Saved translated dataset to {args.output_csv}")


if __name__ == "__main__":
    main()
