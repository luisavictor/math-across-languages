import json
import os

LOGFILE = "/home/iailab76/victorl0/pycharm_sync/gsm8k_debug_samples.jsonl"

def save_output(doc, pred, target):
    """
    LM-Eval postprocess_fn hook.
    Called ONCE per sample.
    Logs model output, question, and target to a jsonl file.
    """
    rec = {
        "question": doc.get("question"),
        "target": target,
        "generated": pred,
    }

    # Append to logfile
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Must return pred unchanged
    return pred
