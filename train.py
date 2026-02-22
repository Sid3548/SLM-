from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from slm.config import load_config
from slm.data import PackedTokenDataset, build_tokenizer, tokenize_to_npy
from slm.model import TinyLM



def get_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32



def cosine_lr(step: int, warmup: int, max_steps: int, lr: float, min_lr: float):
    if step < warmup:
        return lr * step / max(1, warmup)
    t = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + torch.cos(torch.tensor(torch.pi * t)).item())



def train_step(model, x, y, optimizer, scaler, grad_accum_steps, device, dtype, max_grad_norm):
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for _ in range(grad_accum_steps):
        with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
            _, loss = model(x, y)
            loss = loss / grad_accum_steps
        scaler.scale(loss).backward()
        running_loss += loss.item()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    return running_loss



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_12288.yaml")
    parser.add_argument("--build-data", action="store_true")
    parser.add_argument("--build-data-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=0, help="If >0, run epoch-based training")
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = get_dtype(cfg.train.precision)

    if args.build_data or args.build_data_only:
        print("Building tokenizer and tokenized dataset...")
        tok = build_tokenizer(cfg.data, vocab_size=cfg.model.vocab_size)
        tokenize_to_npy(cfg.data, tok)
        if args.build_data_only:
            print("Build-data-only mode complete.")
            return

    dataset = PackedTokenDataset(cfg.data.tokenized_path, seq_len=cfg.model.max_seq_len)
    if len(dataset) == 0:
        raise ValueError("PackedTokenDataset is empty. Build more tokens or reduce max_seq_len.")

    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)

    model = TinyLM(cfg.model).to(device)
    if cfg.train.compile:
        model = torch.compile(model)

    params_m = model.num_params / 1e6
    print(f"Model params: {params_m:.2f}M")
    if params_m < cfg.train.min_params_million:
        raise ValueError(
            f"Model too small: {params_m:.2f}M < required {cfg.train.min_params_million:.2f}M. "
            "Increase d_model/n_layers/ffn_hidden."
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        betas=cfg.train.betas,
        weight_decay=cfg.train.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16 and device == "cuda"))
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    model.train()

    if args.epochs > 0:
        steps_per_epoch = math.ceil(len(loader) / max(1, cfg.train.grad_accum_steps))
        total_steps = max(1, steps_per_epoch * args.epochs)
        global_step = 0
        last_loss = 0.0
        for epoch in range(1, args.epochs + 1):
            epoch_bar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
            micro = 0
            running_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            for x, y in epoch_bar:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                    _, loss = model(x, y)
                    loss = loss / cfg.train.grad_accum_steps
                scaler.scale(loss).backward()
                running_loss += loss.item()
                micro += 1

                if micro % cfg.train.grad_accum_steps == 0:
                    global_step += 1
                    lr = cosine_lr(
                        global_step,
                        cfg.train.warmup_steps,
                        total_steps,
                        cfg.train.lr,
                        cfg.train.min_lr,
                    )
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    last_loss = running_loss
                    running_loss = 0.0
                    if global_step % cfg.train.log_every == 0:
                        epoch_bar.set_postfix(loss=f"{last_loss:.4f}", lr=f"{lr:.2e}")

            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "last_loss": last_loss,
            }
            torch.save(ckpt, save_dir / f"epoch_{epoch}.pt")
            print(f"Epoch {epoch} complete. last_loss={last_loss:.4f}, global_step={global_step}")
        return

    itr = iter(loader)
    pbar = tqdm(range(1, cfg.train.max_steps + 1), desc="train")
    for step in pbar:
        lr = cosine_lr(step, cfg.train.warmup_steps, cfg.train.max_steps, cfg.train.lr, cfg.train.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for _ in range(cfg.train.grad_accum_steps):
            try:
                x, y = next(itr)
            except StopIteration:
                itr = iter(loader)
                x, y = next(itr)

            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                _, loss = model(x, y)
                loss = loss / cfg.train.grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if step % cfg.train.log_every == 0:
            pbar.set_postfix(loss=f"{running_loss:.4f}", lr=f"{lr:.2e}")

        if step % cfg.train.checkpoint_every == 0:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            torch.save(ckpt, save_dir / f"step_{step}.pt")


if __name__ == "__main__":
    main()
