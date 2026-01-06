"""
Translate a seeded 5k sample from the RACE train and test splits to German.

This script mirrors the existing RACE translators but:
- pulls directly from the raw RACE directory structure
- samples a reproducible subset (default 5k) from train and test
- writes two CSVs: race_de_train.csv and race_de_test.csv

The CSVs keep the same schema expected by lm-eval's race_de task:
    columns: article (str), problems (list[dict] as a string), qa (str)
The `problems` field stores question/answer/options in German; the answer
letter is preserved so utils.process_docs can recover the correct target.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL_NAME = "facebook/nllb-200-3.3B"
TARGET_LANG_CODE = "hin_Deva"
DEFAULT_SAMPLE_SIZE = 5000


def split_text_into_chunks(text: str, hard_limit: int = 450) -> List[str]:
    """
    Split text into sentences while keeping end punctuation; fall back to comma chunks for very long sentences.
    This preserves punctuation between sentences when rejoining translated chunks.
    """
    import re

    if not text:
        return [""]
    if len(text) <= hard_limit:
        return [text]

    # Capture sentences WITH their trailing punctuation, plus a fallback for any trailing fragment.
    sentences = re.findall(r"[^.!?]+[.!?]+|\S[^.!?]*$", text)
    chunks: List[str] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > hard_limit:
            # Split long sentences on commas/semicolons/colons, keeping delimiters where possible.
            parts = re.split(r"([,;:])", sentence)
            current = ""
            for part in parts:
                if part is None or part == "":
                    continue
                candidate = (current + part).strip() if current else part.strip()
                if len(candidate) > hard_limit and current:
                    chunks.append(current.strip())
                    current = part.strip()
                else:
                    current = candidate
            if current:
                chunks.append(current.strip())
        else:
            chunks.append(sentence)

    return chunks or [text]


def translate_chunk(
    text: str,
    tokenizer,
    model,
    forced_bos_token_id: int,
    max_input: int = 768,
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
    device = getattr(model, "device", None) or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    encoded = encoded.to(device)

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


def load_model(
    model_name: str,
    target_lang: str,
    hf_token: Optional[str],
    load_in_8bit: bool,
    device_preference: str,
):
    tokenizer_kwargs = {"use_fast": True, "src_lang": "eng_Latn"}
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


def load_race_split(split_root: Path) -> List[Dict]:
    """Load all RACE .txt JSON files under a split (train/test)."""
    examples: List[Dict] = []
    for txt_file in sorted(split_root.rglob("*.txt")):
        try:
            data = json.loads(txt_file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Skipping {txt_file.name}: failed to parse ({exc})")
            continue

        questions = data.get("questions", [])
        options = data.get("options", [])
        answers = data.get("answers", [])
        if not (len(questions) == len(options) == len(answers)):
            print(f"Skipping {txt_file.name}: mismatched QA lengths")
            continue

        examples.append(
            {
                "id": data.get("id", txt_file.stem),
                "article": str(data.get("article", "")).strip(),
                "questions": [str(q).strip() for q in questions],
                "options": [[str(opt).strip() for opt in opt_list] for opt_list in options],
                "answers": [str(ans).strip() for ans in answers],
            }
        )
    return examples


def build_qa_text(article_hi: str, problems: Sequence[Dict]) -> str:
    blocks = [f"कृपया निम्नलिखित पाठ पढ़ें और प्रश्नों के उत्तर दें।\n\n{article_hi}"]
    for idx, problem in enumerate(problems, 1):
        opts = problem.get("options", [])
        opts_serialized = "[" + ", ".join(opts) + "]"
        blocks.append(
            f"प्रश्न {idx}: {problem.get('question', '')}\n"
            f"विकल्प: {opts_serialized}\n"
            f"उत्तर: {problem.get('answer', '')}"
        )
    return "\n\n".join(blocks)


def sample_examples(
    examples: List[Dict], sample_size: int, rng: random.Random
) -> List[Dict]:
    if sample_size >= len(examples):
        return examples
    return rng.sample(examples, sample_size)


def translate_split(
    examples: Iterable[Dict],
    tokenizer,
    model,
    forced_bos_token_id: int,
    output_path: Path,
    save_every: int,
):
    rows = []
    progress = tqdm(examples, total=len(examples), desc=f"Translating -> {output_path.name}")
    for idx, example in enumerate(progress):
        article_de = translate_text(example["article"], tokenizer, model, forced_bos_token_id)

        problems: List[Dict] = []
        for question, opts, answer in zip(
            example["questions"], example["options"], example["answers"]
        ):
            q_de = translate_text(question, tokenizer, model, forced_bos_token_id)
            opts_de = [translate_text(opt, tokenizer, model, forced_bos_token_id) for opt in opts]
            problems.append({"question": q_de, "answer": answer, "options": opts_de})

        qa_text = build_qa_text(article_de, problems)
        rows.append(
            {
                "id": example["id"],
                "article": article_de,
                "problems": json.dumps(problems, ensure_ascii=False),
                "qa": qa_text,
            }
        )

        if save_every and (idx + 1) % save_every == 0:
            pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")
            progress.set_postfix_str(f"saved {idx + 1}")

    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")


def main():
    repo_root = Path(__file__).resolve().parents[2]
    default_data_root = repo_root / "custom_datasets" / "Race_German" / "RACE"

    print(default_data_root)
    default_output_dir = repo_root / "custom_datasets" / "Race_Hindi"

    parser = argparse.ArgumentParser(description="Translate a seeded 5k sample of RACE to German.")
    parser.add_argument("--data_root", type=Path, default=default_data_root, help="Path to RACE root containing train/test folders.")
    parser.add_argument("--output_dir", type=Path, default=default_output_dir, help="Directory for output CSVs.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--hf_token", type=str, default=None, help="HF token if the model is gated.")
    parser.add_argument("--no_8bit", action="store_true", help="Disable 8-bit loading.")
    parser.add_argument("--device", type=str, default="auto", help="Device map: auto, cuda, or cpu.")
    parser.add_argument("--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Rows to sample from each split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--save_every", type=int, default=50, help="Save partial CSV every N rows.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    tokenizer, model, forced_id = load_model(
        args.model_name,
        TARGET_LANG_CODE,
        args.hf_token,
        load_in_8bit=not args.no_8bit,
        device_preference=args.device,
    )

    train_examples = load_race_split(args.data_root / "train")
    test_examples = load_race_split(args.data_root / "test")

    sampled_train = sample_examples(train_examples, args.sample_size, rng)
    sampled_test = sample_examples(test_examples, args.sample_size, rng)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.output_dir / "race_hi_train.csv"
    test_out = args.output_dir / "race_hi_test.csv"

    print(f"Loaded {len(train_examples)} train and {len(test_examples)} test rows.")
    print(f"Translating {len(sampled_train)} train rows -> {train_out}")
    translate_split(sampled_train, tokenizer, model, forced_id, train_out, args.save_every)

    print(f"Translating {len(sampled_test)} test rows -> {test_out}")
    translate_split(sampled_test, tokenizer, model, forced_id, test_out, args.save_every)

    print("Done.")


if __name__ == "__main__":
    main()
