def process_docs(dataset):
    def _clean(doc):
        prompt = str(doc.get("prompt", "")).strip()
        completion = str(doc.get("completion", "")).strip()
        return {
            "prompt": prompt,
            "completion": completion,
        }

    return dataset.map(_clean)

# Compute ROUGE-L F1 for a single prediction/target pair
def process_results(doc, results):
    from rouge_score import rouge_scorer

    prediction = str(results[0]).strip()
    target = str(doc.get("completion", "")).strip()

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(target, prediction)["rougeL"].fmeasure

    return {"rougeL": score}
