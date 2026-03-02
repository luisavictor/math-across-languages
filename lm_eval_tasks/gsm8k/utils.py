import re


def _extract_question_from_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if t.startswith("Q:"):
        t = t[2:].strip()
    split_marker = "\nA:"
    if split_marker in t:
        t = t.split(split_marker, 1)[0].strip()
    return t


def _extract_answer_number(text: str):
    if not isinstance(text, str):
        return None
    t = text.strip()

    m = re.search(r"####\s*([-+]?[0-9]+(?:[.,][0-9]+)?)", t)
    if not m:
        m = re.search(r"The answer is\s*([-+]?[0-9]+(?:[.,][0-9]+)?)", t, flags=re.IGNORECASE)
    if not m:
        return None

    raw = m.group(1).replace(",", ".").strip().rstrip(".")
    try:
        return int(float(raw))
    except Exception:
        return None


def process_docs(dataset):
    def _clean(doc):
        # supports train CSV schema with columns: instruct, qa
        instruct = doc.get("instruct", "")
        qa = doc.get("qa", "")

        question = _extract_question_from_text(instruct if instruct else qa)
        answer_number = _extract_answer_number(qa if qa else instruct)

        return {
            "question": question,
            "answer_number": answer_number,
        }

    return dataset.map(_clean)
