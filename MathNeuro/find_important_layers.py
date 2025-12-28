'''
select_layer_subset computes a subset of layers of the LLM passed as hyperparameter that when pruning the particular layer:
- show a strong drop in performance on gsm8k_cot
- show a weak drop in performance on race
Consequently, this script could help us find math-specific layers that we could use first in order to find math-specific parameters.
For example, given the above hyperparameters, this script extracts layers 6,8,9, and 11 as math-specific.
'''

import argparse
from contextlib import contextmanager
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import lm_eval
import lm_eval.api.registry
from lm_eval.tasks import TaskManager
import numpy as np


@contextmanager
def bypass_layer(layer):
    def hook(_m, inp, out):
        x = inp[0]
        if isinstance(out, torch.Tensor):
            return x
        if isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
            out = list(out)
            out[0] = x
            for i in range(1, len(out)):
                if isinstance(out[i], (tuple, list)) and out[i] and all(isinstance(t, torch.Tensor) for t in out[i]):
                    out[i] = None
            return tuple(out)
        return out

    h = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def gsm8k_acc(lm, tm, limit, bs, gen_kwargs, metric):
    r = lm_eval.simple_evaluate(
        model=lm,
        tasks=["gsm8k_cot"],
        task_manager=tm,
        limit=limit,
        batch_size=bs,
        gen_kwargs=gen_kwargs,
        log_samples=False,
    )
    return float(r["results"]["gsm8k_cot"][metric])


def race_acc(lm, tm, limit, bs, gen_kwargs):
    r = lm_eval.simple_evaluate(
        model=lm,
        tasks=["race"],
        task_manager=tm,
        limit=limit,
        batch_size=bs,
        gen_kwargs=gen_kwargs,
        log_samples=False,
    )
    return float(r["results"]["race"]["acc,none"])


def select_layer_subset(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    limit=3,
    batch_size=1,
    dtype="bfloat16",
    device_map="auto",
    metric="exact_match,strict-match",
    gen_kwargs="do_sample=False,temperature=0.0,top_p=1.0",
    lam=0.5,
    top_frac=0.3,
    include_path="../lm_eval_tasks",
    verbose=True,
):
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, torch_dtype=dtype_map[dtype]
    ).eval()

    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False

    layers = model.model.layers

    hf_cls = lm_eval.api.registry.get_model("hf")
    lm = hf_cls.create_from_arg_obj(
        {"pretrained": model, "tokenizer": tok, "dtype": str(model.dtype).replace("torch.", ""), "use_cache": False},
        {"batch_size": batch_size},
    )

    tm = TaskManager(include_path=include_path)

    with torch.inference_mode():
        base_train = gsm8k_acc(lm, tm, limit, batch_size, gen_kwargs, metric)
        base_cal = race_acc(lm, tm, limit, batch_size, gen_kwargs)

    drop_train, drop_cal = [], []
    for i, layer in enumerate(layers):
        with bypass_layer(layer), torch.inference_mode():
            acc_train = gsm8k_acc(lm, tm, limit, batch_size, gen_kwargs, metric)
            acc_cal = race_acc(lm, tm, limit, batch_size, gen_kwargs)
        drop_train.append(base_train - acc_train)
        drop_cal.append(base_cal - acc_cal)
        if verbose:
            print(f"layer {i:02d} train={acc_train:.6f} drop={drop_train[-1]:.6f} | cal={acc_cal:.6f} drop={drop_cal[-1]:.6f}")

    dtr, dcal = np.array(drop_train), np.array(drop_cal)
    score = dtr - lam * np.maximum(dcal, 0.0)
    k = max(1, int(top_frac * len(dtr)))
    selected = np.argsort(-score)[:k].tolist()

    if verbose:
        print(f"baseline train({metric})={base_train:.6f} baseline cal(acc,none)={base_cal:.6f}")
        print("selected:", selected)

    return {
        "selected_layers": selected,
        "score": score.tolist(),
        "drop_train_task": drop_train,
        "drop_calibration_task": drop_cal,
        "baseline_train": float(base_train),
        "baseline_calibration": float(base_cal),
        "lam": lam,
        "top_frac": top_frac
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--metric", default="exact_match,strict-match")
    ap.add_argument("--gen_kwargs", default="do_sample=False,temperature=0.0,top_p=1.0")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--top_frac", type=float, default=0.3)
    ap.add_argument("--include_path", default="../lm_eval_tasks")
    args = ap.parse_args()

    # call with defaults (and CLI overrides if provided)
    select_layer_subset(
        model_name=args.model,
        limit=args.limit,
        batch_size=args.batch_size,
        dtype=args.dtype,
        device_map=args.device_map,
        metric=args.metric,
        gen_kwargs=args.gen_kwargs,
        lam=args.lam,
        top_frac=args.top_frac,
        include_path=args.include_path,
        verbose=True,
    )


if __name__ == "__main__":
    main()
