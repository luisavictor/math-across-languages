import json
import functools

LOGFILE = "model_outputs.jsonl"

def patch_hf_generate_until(HF_LM):
    original = HF_LM.generate_until

    @functools.wraps(original)
    def wrapped(self, requests):
        outputs = original(self, requests)

        with open(LOGFILE, "a", encoding="utf8") as f:
            for req, out in zip(requests, outputs):
                # for generate_until: out is a list of strings
                txt = out[0] if isinstance(out, list) else str(out)

                f.write(json.dumps({
                    "prompt": req.arguments[0],
                    "output": txt,
                    "doc_id": req.doc_id if hasattr(req, "doc_id") else None
                }, ensure_ascii=False) + "\n")

        return outputs

    HF_LM.generate_until = wrapped
