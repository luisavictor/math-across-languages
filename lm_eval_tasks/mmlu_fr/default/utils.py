import ast


def _parse_choices(raw_choices):
    if isinstance(raw_choices, list):
        return [str(choice) for choice in raw_choices]
    if raw_choices is None:
        return []
    if not isinstance(raw_choices, str):
        try:
            return [str(choice) for choice in list(raw_choices)]
        except TypeError:
            return [str(raw_choices)]

    try:
        parsed = ast.literal_eval(raw_choices)
        if isinstance(parsed, dict):
            return [str(value) for value in parsed.values()]
        return [str(value) for value in parsed]
    except (ValueError, SyntaxError):
        cleaned = raw_choices.strip().strip("[]")
        if not cleaned:
            return []
        parts = [part.strip(" '\"") for part in cleaned.split(",")]
        return [part for part in parts if part]


def process_docs(dataset):
    def _clean(doc):
        return {"choices": _parse_choices(doc.get("choices"))}

    return dataset.map(_clean)

