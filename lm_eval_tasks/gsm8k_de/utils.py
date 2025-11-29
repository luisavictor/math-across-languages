import re

def process_docs(dataset):
    def _clean(doc):
        instruct = doc["instruct"].strip()

        # 1. Extract the question (everything between 'F: ' and ' A:')
        # Example: "F: ... A: Denken wir ..."
        q_match = re.search(r'^F:\s*(.*?)\s*A:', instruct, flags=re.DOTALL)
        question = q_match.group(1).strip() if q_match else instruct

        # 2. Extract final answer from #### NUMBER
        # Example: "#### 72"
        a_match = re.search(r'####\s*([0-9]+)', instruct)
        if a_match:
            answer = a_match.group(1).strip()
        else:
            # fallback: try extracting from "Die Antwort ist X."
            qa = doc.get("qa", "")
            a2_match = re.search(r'Antwort ist\s*([0-9]+)', qa)
            answer = a2_match.group(1).strip() if a2_match else ""

        # 3. Return clean GSM8K-like structure
        return {
            "question": question,
            "answer": answer,
        }

    return dataset.map(_clean)

