import re


def _extract_question(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    t = re.sub(r"^(Q|प्रश्न)\s*:\s*", "", t)
    if "\nA:" in t:
        t = t.split("\nA:", 1)[0].strip()
    return t


def _extract_answer_number(text: str):
    if not isinstance(text, str):
        return None
    t = text.strip()

    m = re.search(r"####\s*([-+]?[0-9]+(?:[.,][0-9]+)?)", t)
    if not m:
        m = re.search(r"(उत्तर है|उत्तर)\s*([-+]?[0-9]+(?:[.,][0-9]+)?)\s*है?", t)
    if not m:
        m = re.search(r"The answer is\s*([-+]?[0-9]+(?:[.,][0-9]+)?)", t, flags=re.IGNORECASE)
        if m:
            num = m.group(1)
        else:
            return None
    else:
        num = m.group(1 if "####" in m.re.pattern else 2)

    num = num.replace(",", ".").strip().rstrip(".")
    try:
        return int(float(num))
    except Exception:
        return None


def process_docs(dataset):
    def _clean(doc):
        instruct = (doc.get("instruct") or "").strip()
        qa = (doc.get("qa") or "").strip()
        question = _extract_question(instruct if instruct else qa)
        answer_number = _extract_answer_number(qa if qa else instruct)
        return {"question": question, "answer_number": answer_number}

    return dataset.map(_clean)
