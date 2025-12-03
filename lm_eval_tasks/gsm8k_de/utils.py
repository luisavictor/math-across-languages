import re
'''
def process_docs(dataset):
    def _clean(doc):
        instruct = doc["instruct"].strip()
        qa = doc.get("qa", "").strip()

        # ---- Extract question between F: and A: ----
        # Example instruct:
        # "F: ...? A: Denken wir Schritt für Schritt. ..."
        q_match = re.search(r"F:\s*(.*?)\s*A:", instruct, flags=re.DOTALL)
        if q_match:
            question = q_match.group(1).strip()
        else:
            # Fallback: use full 'instruct' if pattern fails
            question = instruct

        # ---- Extract final numeric answer ----
        # 1) prefer the '#### 72' style in 'instruct'
        a_match = re.search(r"####\s*([0-9\.,\-]+)", instruct)
        if a_match:
            num_str = a_match.group(1)
        else:
            # 2) fallback: "Die Antwort ist 72." in 'qa'
            a2_match = re.search(r"(Die Antwort ist|Antwort ist)\s*([0-9\.,\-]+)", qa)
            if a2_match:
                num_str = a2_match.group(2)
            else:
                num_str = ""

        # normalize "72." / "72," / "72,0" -> string number
        num_str = num_str.replace(",", ".").strip().rstrip(".")

        # convert to int if possible, else keep string
        try:
            answer_number = int(float(num_str))
        except Exception:
            answer_number = num_str  # fallback, still works as string target

        return {
            "question": question,
            "answer_number": answer_number,
        }

    return dataset.map(_clean)
'''



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
            "answer_number": answer_number,  # <-- always int or None
        }

    return dataset.map(_clean)
