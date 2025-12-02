import re

def process_docs(dataset):
    def _clean(doc):
        instruct = doc["instruct"].strip()

        # ---- Extract question between F: and A: ----
        q_match = re.search(r"F:\s*(.*?)\s*A:", instruct, flags=re.DOTALL)
        question = q_match.group(1).strip() if q_match else instruct

        # ---- Extract final numeric answer from #### ----
        a_match = re.search(r"####\s*([0-9\.,\-]+)", instruct)
        if a_match:
            answer = a_match.group(1).replace(",", ".").strip()
        else:
            # fallback: maybe inside qa field ("Die Antwort ist X.")
            qa = doc.get("qa", "")
            a2_match = re.search(r"Antwort ist\s*([0-9\.,\-]+)", qa)
            answer = a2_match.group(1).replace(",", ".").strip() if a2_match else ""

        return {
            "question": question,
            "answer": answer,
        }

    return dataset.map(_clean)
