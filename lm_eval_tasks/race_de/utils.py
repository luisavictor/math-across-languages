import ast

# Map answer letters to list indices
LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


def _answer_to_index(raw_answer, num_choices):
    """Normalize answer field (letter or index) to a safe integer index."""
    idx = None
    if isinstance(raw_answer, str):
        value = raw_answer.strip().upper()
        if value in LETTER_TO_INDEX:
            idx = LETTER_TO_INDEX[value]
        elif value.isdigit():
            idx = int(value)
    elif isinstance(raw_answer, int):
        idx = raw_answer

    if idx is None:
        return None
    if 0 <= idx < num_choices:
        return idx
    return None


def process_docs(dataset):
    """Flatten the Race-style CSV so each question becomes its own row."""

    def _process(batch):
        articles, questions, options_list, targets = [], [], [], []

        for article, problems in zip(batch["article"], batch["problems"]):
            try:
                parsed = ast.literal_eval(problems)
            except Exception:
                parsed = []

            if not isinstance(parsed, list):
                continue

            for problem in parsed:
                opts = problem.get("options", [])
                if not isinstance(opts, list) or not opts:
                    continue

                answer_idx = _answer_to_index(problem.get("answer"), len(opts))
                if answer_idx is None:
                    continue

                articles.append(article)
                questions.append(str(problem.get("question", "")).strip())
                cleaned_opts = [str(option).strip() for option in opts]
                options_list.append(cleaned_opts)
                targets.append(cleaned_opts[answer_idx])

        return {
            "article": articles,
            "question": questions,
            "options": options_list,
            "target": targets,
        }

    return dataset.map(_process, batched=True, remove_columns=dataset.column_names)
