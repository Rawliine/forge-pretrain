"""
Starter training script for the gpu-mode Paris hackathon training track
"""

import os
import time
import glob
import math
import argparse
from contextlib import nullcontext
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from model import get_model


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Data
    data_dir:    str   = "data"
    token_dtype: str   = "uint16"
    seq_len:     int   = 1024

    # Model (passed through to get_model — add arch-specific keys in model.py)
    vocab_size: int   = 32768
    n_layer:    int   = 12
    n_head:     int   = 12
    n_embd:     int   = 768
    dropout:    float = 0.0

    # Training
    batch_size:       int   = 8
    grad_accum_steps: int   = 8
    muon_lr:          float = 0.02
    adam_lr:           float = 3e-4
    warmup_steps:     int   = 100
    max_steps:        int   = 10_000
    weight_decay:     float = 0.01
    grad_clip:        float = 1.0
    time_limit_seconds: float = 10 * 60

    # Checkpointing
    checkpoint_path: str = "checkpoint.pt"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BinDataset:
    """Memory-maps all *.bin files and draws random (seq_len+1)-token windows."""

    def __init__(self, data_dir: str, seq_len: int, dtype: str = "uint16"):
        paths = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        if not paths:
            raise FileNotFoundError(f"No *.bin files found in '{data_dir}'")
        self.seq_len  = seq_len
        np_dtype      = np.dtype(dtype)
        self.shards   = [np.memmap(p, dtype=np_dtype, mode="r") for p in paths]
        self.lengths  = [len(s) for s in self.shards]
        self.total    = sum(self.lengths)
        self.weights  = [l / self.total for l in self.lengths]
        print(f"[data] {len(paths)} shard(s), {self.total:,} tokens total")

    def get_batch(self, batch_size: int, device):
        xs, ys = [], []
        for _ in range(batch_size):
            shard = self.shards[np.random.choice(len(self.shards), p=self.weights)]
            start = np.random.randint(0, len(shard) - self.seq_len - 1)
            chunk = torch.from_numpy(shard[start:start + self.seq_len + 1].astype(np.int64))
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        return torch.stack(xs).to(device), torch.stack(ys).to(device)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay → min_lr
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: Config, max_lr: float) -> float:
    min_lr = max_lr * 0.1
    if step < cfg.warmup_steps:
        return max_lr * step / cfg.warmup_steps
    if step >= cfg.max_steps:
        return min_lr
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_lr - min_lr)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, step: int, cfg: Config):
    raw_model = model.module if hasattr(model, "module") else model
    torch.save({
        "step":   step,
        "model":  raw_model.state_dict(),
        "config": asdict(cfg),
    }, cfg.checkpoint_path)
    print(f"[ckpt] saved → {cfg.checkpoint_path}  (step {step})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",          default="data")
    parser.add_argument("--checkpoint_path",   default="checkpoint.pt")
    parser.add_argument("--seq_len",           type=int,   default=1024)
    parser.add_argument("--vocab_size",        type=int,   default=32768)
    parser.add_argument("--n_layer",           type=int,   default=12)
    parser.add_argument("--n_head",            type=int,   default=12)
    parser.add_argument("--n_embd",            type=int,   default=768)
    parser.add_argument("--batch_size",        type=int,   default=8)
    parser.add_argument("--grad_accum_steps",  type=int,   default=4)
    parser.add_argument("--max_steps",         type=int,   default=10_000)
    parser.add_argument("--time_limit_min",    type=float, default=10.0)
    parser.add_argument("--fp16_compress",     action="store_true",
                        help="FP16 gradient compression for DDP AllReduce")
    args = parser.parse_args()

    cfg = Config(
        data_dir           = args.data_dir,
        checkpoint_path    = args.checkpoint_path,
        seq_len            = args.seq_len,
        vocab_size         = args.vocab_size,
        n_layer            = args.n_layer,
        n_head             = args.n_head,
        n_embd             = args.n_embd,
        batch_size         = args.batch_size,
        grad_accum_steps   = args.grad_accum_steps,
        max_steps          = args.max_steps,
        time_limit_seconds = args.time_limit_min * 60,
    )

    # ------------------------------------------------------------------ DDP
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        rank       = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        device     = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master     = rank == 0
    else:
        rank = 0; master = True
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(1337 + rank)
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) \
              if "cuda" in device else nullcontext()
    scaler = torch.cuda.amp.GradScaler() if "cuda" in device else None
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True

    # ------------------------------------------------------------------ Model
    model = get_model(asdict(cfg)).to(device)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        model = torch.compile(model)
    elif master:
        print(f"[compile] skipping torch.compile (device capability "
              f"{torch.cuda.get_device_capability()} < 7.0)")
    if "cuda" in device:
        torch.cuda.synchronize(device)
        m_params = torch.cuda.memory_allocated() / 1e6
        print(f"[mem] rank={rank} after model load: {m_params:.0f} MB  (≈ M_params)")
    else:
        m_params = 0.0
    if master:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] {n_params/1e6:.1f}M parameters")

    if ddp:
        model = DDP(model, device_ids=[local_rank])
        if args.fp16_compress:
            from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
            model.register_comm_hook(state=None, hook=fp16_compress_hook)
            if master:
                print("[ddp] fp16 gradient compression enabled")

    # ------------------------------------------------------------------ Optimizer
    # Muon for 2D hidden-layer weights; AdamW for embeddings, LN, biases, lm_head
    raw_model = model.module if ddp else model
    hidden_weights = [p for n, p in raw_model.named_parameters()
                      if p.requires_grad and p.ndim >= 2
                      and 'wte' not in n and 'wpe' not in n and 'lm_head' not in n]
    other_params   = [p for n, p in raw_model.named_parameters()
                      if p.requires_grad
                      and (p.ndim < 2 or 'wte' in n or 'wpe' in n or 'lm_head' in n)]
    if master:
        print(f"[optim] Muon: {len(hidden_weights)} params, "
              f"AdamW: {len(other_params)} params")
    muon_opt = torch.optim.Muon(
        [{"params": hidden_weights}],
        lr=cfg.muon_lr, momentum=0.95, weight_decay=cfg.weight_decay,
    )
    adam_opt = torch.optim.AdamW(
        [{"params": other_params}],
        lr=cfg.adam_lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay,
    )

    # ------------------------------------------------------------------ Data
    dataset = BinDataset(cfg.data_dir, cfg.seq_len, cfg.token_dtype)

    # ------------------------------------------------------------------ Train
    WARMUP_STEPS = 10  # steps before steady-state memory snapshot

    step        = 0
    train_start = time.time()
    model.train()
    muon_opt.zero_grad()
    adam_opt.zero_grad()

    # Per-phase memory collected on the first post-warmup step
    _mem_snapshot_done = False

    while step < cfg.max_steps:

        # Time-limit check — never starts a new step after the deadline
        elapsed = time.time() - train_start
        stop = torch.tensor(int(elapsed >= cfg.time_limit_seconds), device=device)
        if ddp:
            dist.broadcast(stop, src=0)
        if stop.item():
            if master:
                print(f"\n[time] {elapsed/60:.1f} min elapsed — time limit reached.")
                save_checkpoint(model, step, cfg)
            break

        step_start = time.time()
        for pg in muon_opt.param_groups:
            pg["lr"] = get_lr(step, cfg, cfg.muon_lr)
        for pg in adam_opt.param_groups:
            pg["lr"] = get_lr(step, cfg, cfg.adam_lr)

        do_mem = "cuda" in device and step == WARMUP_STEPS and not _mem_snapshot_done

        # ── CUDA event timers ────────────────────────────────────────────────
        if "cuda" in device:
            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end   = torch.cuda.Event(enable_timing=True)
            bwd_end   = torch.cuda.Event(enable_timing=True)
            opt_end   = torch.cuda.Event(enable_timing=True)

        # Gradient accumulation
        accumulated_loss = 0.0
        for micro_step in range(cfg.grad_accum_steps):
            x, y     = dataset.get_batch(cfg.batch_size, device)
            sync_ctx = model.no_sync() if (ddp and micro_step < cfg.grad_accum_steps - 1) \
                       else nullcontext()

            is_last_micro = micro_step == cfg.grad_accum_steps - 1

            if do_mem and micro_step == 0:
                torch.cuda.reset_peak_memory_stats(device)

            if "cuda" in device and is_last_micro:
                fwd_start.record()

            with sync_ctx, amp_ctx:
                _, loss = model(x, y)
                loss    = loss / cfg.grad_accum_steps

            if "cuda" in device and is_last_micro:
                fwd_end.record()

            if do_mem and is_last_micro:
                torch.cuda.synchronize(device)
                m_fwd_peak = torch.cuda.max_memory_allocated(device) / 1e6
                m_fwd_live = torch.cuda.memory_allocated(device) / 1e6
                print(f"[mem] rank={rank} after fwd: peak={m_fwd_peak:.0f} "
                      f"live={m_fwd_live:.0f} MB  "
                      f"activations≈{m_fwd_live - m_params:.0f} MB")
                torch.cuda.reset_peak_memory_stats(device)

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if "cuda" in device and is_last_micro:
                bwd_end.record()

            if do_mem and is_last_micro:
                torch.cuda.synchronize(device)
                m_bwd_peak = torch.cuda.max_memory_allocated(device) / 1e6
                m_bwd_live = torch.cuda.memory_allocated(device) / 1e6
                print(f"[mem] rank={rank} after bwd: peak={m_bwd_peak:.0f} "
                      f"live={m_bwd_live:.0f} MB  "
                      f"gradients≈{m_bwd_live - m_params:.0f} MB")
                torch.cuda.reset_peak_memory_stats(device)

            accumulated_loss += loss.item()

        if cfg.grad_clip > 0:
            if scaler is not None:
                scaler.unscale_(muon_opt)
                scaler.unscale_(adam_opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        if scaler is not None:
            scaler.step(muon_opt)
            scaler.step(adam_opt)
            scaler.update()
        else:
            muon_opt.step()
            adam_opt.step()
        muon_opt.zero_grad(set_to_none=True)
        adam_opt.zero_grad(set_to_none=True)

        if "cuda" in device:
            opt_end.record()

        if do_mem:
            torch.cuda.synchronize(device)
            m_opt_peak = torch.cuda.max_memory_allocated(device) / 1e6
            m_opt_live = torch.cuda.memory_allocated(device) / 1e6
            m_reserved = torch.cuda.max_memory_reserved(device) / 1e6
            print(f"[mem] rank={rank} after opt.step: peak={m_opt_peak:.0f} "
                  f"live={m_opt_live:.0f} MB  "
                  f"opt_state≈{m_opt_live - m_params:.0f} MB  "
                  f"reserved≈{m_reserved:.0f} MB (≈nvidia-smi)")
            _mem_snapshot_done = True

        step_time = time.time() - step_start
        step += 1

        if "cuda" in device:
            torch.cuda.synchronize(device)
            t_fwd = fwd_start.elapsed_time(fwd_end)
            t_bwd = fwd_end.elapsed_time(bwd_end)
            t_opt = bwd_end.elapsed_time(opt_end)
        else:
            t_fwd = t_bwd = t_opt = float("nan")

        tok_per_rank  = cfg.batch_size * cfg.grad_accum_steps * cfg.seq_len
        tok_s_rank    = tok_per_rank / step_time

        if master and step % 10 == 0:
            tok_per_step  = tok_per_rank
            if ddp:
                tok_per_step *= dist.get_world_size()
            tok_s         = tok_per_step / step_time
            elapsed_total = time.time() - train_start
            remaining     = max(0, cfg.time_limit_seconds - elapsed_total)
            print(f"step={step} | loss={accumulated_loss:.4f} | "
                  f"lr(muon)={get_lr(step, cfg, cfg.muon_lr):.2e} "
                  f"lr(adam)={get_lr(step, cfg, cfg.adam_lr):.2e} | "
                  f"tok/s={tok_s:.0f} (rank tok/s={tok_s_rank:.0f}) | "
                  f"{step_time*1000:.0f}ms/step "
                  f"[fwd={t_fwd:.0f} bwd={t_bwd:.0f} opt={t_opt:.0f} ms] | "
                  f"elapsed {elapsed_total/60:.1f}m | "
                  f"time left {remaining/60:.1f}m")

    # max_steps reached cleanly
    if step >= cfg.max_steps and master:
        print(f"\n[done] Reached max_steps={cfg.max_steps}.")
        save_checkpoint(model, step, cfg)

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()

