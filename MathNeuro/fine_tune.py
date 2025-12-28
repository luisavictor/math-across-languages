import math
import random
from collections import deque
from contextlib import nullcontext
import torch


def fine_tune_on_isolated_params(
    model,
    tokenizer,
    train_df,
    isolated_masks,
    num_epochs=4,
    lr=5e-5,
    max_length=512,
    use_amp=True,
    amp_dtype=torch.bfloat16,
    grad_clip=1.0,
    weight_decay=1e-3,
    store_base_fp32=True,
    grad_accum_steps=16,
    warmup_ratio=0.1,
    log_every=30,
    check_drift_every=300,
    drift_eps=1e-8,
    seed=None,
    cudnn_benchmark=False,
):
    device = next(model.parameters()).device
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            import numpy as np
        except Exception:
            np = None
        if np is not None:
            np.random.seed(seed)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # Build trainable set + cache masks + base weights
    trainables = []
    mask_cache = {}
    base_cache = {}

    for name, p in model.named_parameters():
        if name in isolated_masks:
            m = isolated_masks[name]
            if m.dtype == torch.bool:
                m = m.to(torch.float32)

            if torch.count_nonzero(m).item() > 0:
                p.requires_grad = True
                trainables.append((name, p))

                mask_cache[name] = m.to(device=device, dtype=p.dtype)

                if store_base_fp32:
                    base_cache[name] = p.detach().to(device=device, dtype=torch.float32).clone()
                else:
                    base_cache[name] = p.detach().to(device=device).clone()
            else:
                p.requires_grad = False
        else:
            p.requires_grad = False

    if not trainables:
        print("No isolated task-specific parameters found. Skipping fine-tuning.")
        return

    # Gradient masking hooks
    hooks = []
    for name, p in trainables:
        m = mask_cache[name]
        hooks.append(p.register_hook(lambda g, m=m: g.mul(m)))

    optimizer = torch.optim.AdamW(
        [p for _, p in trainables],
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
        foreach=False,
    )

    total_steps = num_epochs * len(train_df)
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    model.train()
    global_step = 0
    opt_step = 0  # optimizer steps (after accumulation)



    def split_qa_cot(qa_text: str):
        # target = cot + The answer is XY
        if "A:" in qa_text:
            question_part, _, answer_part = qa_text.partition("A:")
            prompt_text = question_part.strip()
            if not prompt_text.endswith("A:"):
                prompt_text = prompt_text.rstrip() + "\nA:"
            return prompt_text, answer_part.lstrip()
        #return qa_text.strip(), ""

        else:
             prompt_part = qa_text["question"]
             answer_part = qa_text["solution"]
             return prompt_part, answer_part




    def set_lr(step_idx: int):
        if step_idx <= warmup_steps:
            lr_scale = step_idx / warmup_steps
        else:
            t = (step_idx - warmup_steps) / max(1, (total_steps - warmup_steps))
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * t))
        new_lr = lr * lr_scale
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr
        return new_lr

    def projection():
        with torch.no_grad():
            for name, p in trainables:
                m = mask_cache[name]
                base = base_cache[name]
                base_cast = base.to(dtype=p.dtype) if base.dtype != p.dtype else base
                p.data.mul_(m).add_(base_cast * (1 - m))

    def check_drift():
        # probably, we dont need this, just for checking that no undesired params are updated/changed --> would print warning then
        with torch.no_grad():
            max_drift = 0.0
            worst = None
            for name, p in trainables:
                m = mask_cache[name]
                base = base_cache[name]
                base_cast = base.to(dtype=p.dtype) if base.dtype != p.dtype else base
                nonmask = (1 - m)
                if torch.count_nonzero(nonmask).item() == 0:
                    continue
                drift = torch.max(torch.abs((p - base_cast) * nonmask)).item()
                if drift > max_drift:
                    max_drift = float(drift)
                    worst = name
            if max_drift > drift_eps:
                print(f"[WARN] non-masked drift={max_drift:.3e} worst={worst}")
            return max_drift


    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        if seed is None:
            epoch_df = train_df.sample(frac=1).reset_index(drop=True)
        else:
            epoch_df = train_df.sample(frac=1, random_state=seed + epoch).reset_index(drop=True)

        for _, row in epoch_df.iterrows():
            global_step += 1
            cur_lr = set_lr(global_step)
            if "qa" in row:
                qa_text = row["qa"]
            else:
                qa_text = row
            prompt_text, target_text = split_qa_cot(qa_text)
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=True,
                truncation=False,
                max_length=max_length,
            )["input_ids"]
            target_ids = tokenizer(
                target_text,
                add_special_tokens=False,
                truncation=False,
                max_length=max_length,
            )["input_ids"]

            print("prompt text:", prompt_text)
            print("target text:", target_text)

            if len(target_ids) == 0:
                continue

            if len(prompt_ids) + len(target_ids) > max_length:
                max_prompt_len = max_length - len(target_ids)
                if max_prompt_len < 1:
                    if tokenizer.bos_token_id is not None:
                        prompt_ids = [tokenizer.bos_token_id]
                        target_ids = target_ids[-(max_length - 1):]
                    else:
                        prompt_ids = []
                        target_ids = target_ids[-max_length:]
                else:
                    prompt_ids = prompt_ids[-max_prompt_len:]

            input_ids = torch.tensor([prompt_ids + target_ids], device=device)
            labels = torch.tensor([[-100] * len(prompt_ids) + target_ids], device=device)

            tgt_len = len(target_ids)

            autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
            with autocast_ctx:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / grad_accum_steps  # normalize for accumulation

            # logging
            loss_window = deque(maxlen=20)
            loss_window.append(float((loss.detach().item()) * grad_accum_steps))
            if global_step % log_every == 0 or global_step == 1:
                meanN = sum(loss_window) / len(loss_window)
                print(f"step={global_step}/{total_steps} opt_step={opt_step} lr={cur_lr:.3e} "
                      f"tgt_len={tgt_len} loss={loss_window[-1]:.6f} mean(last {len(loss_window)})={meanN:.6f}")

            # backward
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # optimizer step every grad_accum_steps
            if (global_step % grad_accum_steps) == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_([p for _, p in trainables], grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_([p for _, p in trainables], grad_clip)
                    optimizer.step()

                opt_step += 1
                projection()
                optimizer.zero_grad(set_to_none=True)
                if (opt_step % max(1, check_drift_every // max(1, grad_accum_steps))) == 0:
                    check_drift()

            del outputs, input_ids, labels, prompt_ids

    for h in hooks:
        h.remove()
