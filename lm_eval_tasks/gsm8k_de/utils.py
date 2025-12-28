import re

def process_docs(dataset):
    def _clean(doc):

        question = doc["question"].strip()
        answer_text = doc["answer"].strip()

        # ---- Extract numeric answer ----
        a_match = re.search(r"####\s*([0-9\.,\-]+)", answer_text)
        if a_match:
            num_str = a_match.group(1)
        else:
            # fallback detection
            a2_match = re.search(r"(Die Antwort ist|Antwort ist)\s*([0-9\.,\-]+)", answer_text)
            if a2_match:
                num_str = a2_match.group(2)
            else:
                num_str = None    # <-- CRITICAL FIX

        if num_str is not None:
            num_str = num_str.replace(",", ".").strip().rstrip(".")

            try:
                answer_number = int(float(num_str))
            except:
                answer_number = None
        else:
            answer_number = None

        return {
            "question": question,
            "answer_number": answer_number,
        }

    return dataset.map(_clean)
