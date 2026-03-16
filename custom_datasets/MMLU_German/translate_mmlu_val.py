import argparse
import csv
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL_NAME = "facebook/nllb-200-3.3B"
#TARGET_LANG_CODE = "deu_Latn"
TARGET_LANG_CODE = "fra_Latn"


QA_LABELS = {
    "deu_Latn": ("Frage", "Antwortoptionen", "Antwort"),
    "hin_Deva": ("प्रश्न", "उत्तर विकल्प", "उत्तर"),
    "fra_Latn": ("Question", "Choix", "Réponse"),
    "ita_Latn": ("Domanda", "Opzioni di risposta", "Risposta"),
    "por_Latn": ("Pergunta", "Opções de resposta", "Resposta"),
}


ANSWER_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
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



def strip_outer_quotes(text: str) -> str:
    if not isinstance(text, str):
        return text
    cleaned = text.replace('""', '"').replace("''", "'").strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1]
    return cleaned.strip()


def format_choices_mmlu(choices: Sequence[str]) -> str:
    normalized = []
    for choice in choices:
        c = strip_outer_quotes(choice)
        c = c.replace('"', "'")
        normalized.append(c)
    return "[" + ", ".join(f"'{c}'" for c in normalized) + "]"


def split_text_into_chunks(text: str, hard_limit: int = 180) -> List[str]:
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


def normalize_answer(raw_answer: str) -> int:
    if raw_answer is None:
        raise ValueError("Missing answer label.")
    value = str(raw_answer).strip()
    if not value:
        raise ValueError("Empty answer label.")

    upper = value.upper()
    if upper in ANSWER_MAP:
        return ANSWER_MAP[upper]

    try:
        num = int(value)
        if num in {0, 1, 2, 3}:
            return num
        if num in {1, 2, 3, 4}:
            return num - 1
    except Exception:
        pass

    raise ValueError(f"Unrecognized answer label: {raw_answer}")


def load_val_dataset(input_dir: Path, max_rows_per_subject: Optional[int]) -> pd.DataFrame:
    rows = []
    csv_paths = sorted(input_dir.glob("*_val.csv"))
    for csv_path in csv_paths:
        subject = csv_path.name.replace("_val.csv", "")
        if subject in EXCLUDED_SUBJECTS:
            continue
        df = pd.read_csv(
            csv_path,
            header=None,
            names=["question", "A", "B", "C", "D", "answer"],
            nrows=max_rows_per_subject,
            keep_default_na=False,
            quoting=csv.QUOTE_MINIMAL,
        )
        for _, row in df.iterrows():
            try:
                answer_idx = normalize_answer(row["answer"])
            except ValueError as exc:
                print(f"Skipping row with invalid answer in {csv_path.name}: {exc}")
                continue

            choices = [
                strip_outer_quotes(str(row["A"])),
                strip_outer_quotes(str(row["B"])),
                strip_outer_quotes(str(row["C"])),
                strip_outer_quotes(str(row["D"])),
            ]

            rows.append(
                {
                    "question": strip_outer_quotes(str(row["question"])),
                    "subject": subject,
                    "choices": choices,
                    "answer": answer_idx,
                }
            )

    return pd.DataFrame(rows)


def translate_rows(
    df: pd.DataFrame,
    tokenizer,
    model,
    forced_bos_token_id: int,
    save_every: int,
    output_path: Path,
) -> pd.DataFrame:
    translated_rows = []
    progress = tqdm(df.itertuples(index=False), total=len(df), desc="Translating rows", unit="row")
    for idx, row in enumerate(progress, start=1):
        translated_choices = [
            strip_outer_quotes(
                translate_text(choice, tokenizer, model, forced_bos_token_id)
            )
            for choice in row.choices
        ]
        question_de = strip_outer_quotes(
            translate_text(str(row.question), tokenizer, model, forced_bos_token_id)
        )
        qa_de = build_qa_text(question_de, translated_choices, row.answer, TARGET_LANG_CODE)

        translated_rows.append(
            {
                "question": question_de,
                "subject": row.subject,
                "choices": format_choices_mmlu(translated_choices),
                "answer": row.answer,
                "qa": qa_de,
            }
        )

        if save_every and idx % save_every == 0:
            pd.DataFrame(translated_rows).to_csv(output_path, index=False, encoding="utf-8")
            print(f"Saved progress after {idx} rows to {output_path}")

    return pd.DataFrame(translated_rows)


def main():
    script_dir = Path(__file__).resolve().parent
    default_input_dir = script_dir / "data_val"
    default_output = script_dir / ".."/"MMLU_French"/"mmlu_fr_val.csv"

    parser = argparse.ArgumentParser(description="Translate MMLU validation split to French.")
    parser.add_argument("--input_dir", type=Path, default=default_input_dir, help="Directory with *_val.csv files.")
    parser.add_argument("--output_csv", type=Path, default=default_output, help="Output CSV path.")
    parser.add_argument(
        "--max_rows_per_subject",
        type=int,
        default=None,
        help="Optionally limit rows per subject for quick tests.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
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
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Seq2Seq model to use for translation.",
    )
    args = parser.parse_args()

    df = load_val_dataset(args.input_dir, args.max_rows_per_subject)
    if df.empty:
        raise RuntimeError(f"No validation rows loaded from {args.input_dir}")

    tokenizer, model, forced_id = load_model(
        args.model_name,
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
    print(f"Saved translated validation dataset to {args.output_csv}")


if __name__ == "__main__":
    main()