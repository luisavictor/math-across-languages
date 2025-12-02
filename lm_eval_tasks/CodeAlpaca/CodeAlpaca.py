def process_docs(dataset):
    def _clean(doc):
        prompt = str(doc.get("prompt", "")).strip()
        completion = str(doc.get("completion", "")).strip()
        return {
            "prompt": prompt,
            "completion": completion,
        }

    return dataset.map(_clean)
