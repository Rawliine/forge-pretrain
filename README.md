# LLM Training Hackathon

## The Challenge
Train the best language model you can in **10 minutes of GPU time** on a 32-GPU cluster.  
Judged on **validation loss** (perplexity, lower is better). *(HellaSwag accuracy as tiebreaker.)*

---

- Pre-tokenized `uint16` binary shards in `/home/data/chunk*` — vocab size **32,000**

## What You Submit
**Two Python files** (`model.py` and `train.py`) plus an optional `requirements.txt`.

- No embedded binary blobs or external assets
- `requirements.txt` packages are installed before the clock starts
- Any data preprocessing must happen within the training script
- The 10-minute GPU clock starts from the **first forward pass**

We evaluate by running your `train.py` (which must produce `checkpoint.pt`), then loading it with your `model.py`:
```python
ckpt  = torch.load("checkpoint.pt", weights_only=True)
model = get_model(ckpt["config"])   # from your model.py
model.load_state_dict(ckpt["model"])
```

`model.py` must expose `get_model(config: dict) -> nn.Module`. The `config` is whatever
dict you saved — use it to store the hyperparameters needed to reconstruct your architecture.
`forward` must be `(idx, targets=None) -> (logits, loss)`.

---

## Rules
- No external pretrained weights or training data
- Everything else goes — custom kernels, custom optimizers, ensembles, multiple training runs, whatever
- **10 minutes of GPU time**, counted from the first forward pass to last forward pass (i.e., finishing the step-inflight is fine (if this is a few seconds), staring a new one is not)
- SLURM wall time is **12 minutes** to allow for NCCL init and checkpoint saving

---

## Starter Code
`model.py`, `train.py`, and `submit.sh` are provided as a working GPT baseline — DDP, bfloat16,
cosine schedule, and the 10-minute timer built in. **You are free to ignore them entirely**
and bring your own stack, as long as the checkpoint contract above is respected.

---

## Running Experiments

Use `experiment.sh` to run a named experiment, auto-evaluate the checkpoint, and log results:

```bash
./experiment.sh "baseline adamw lr3e-4"
```

This will:
1. Run `run_local.sh` (training) and stream output to `logs/`
2. Evaluate the resulting `checkpoint.pt` via `eval_checkpoint.py` on a held-out shard
3. Append a row to `results.md` with steps, tok/s, train loss, val loss, perplexity, and wall time

**GPU selection**:
```bash
GPUS=0,1,2,3 
```

**Results** are tracked in `results.md` — one row per run, auto-created on first use.

### Heterogeneous-GPU runs (e.g. RTX 3070 Ti + GTX 1070 Ti)

`train.py` uses FSDP `FULL_SHARD` with a `MixedPrecision(param=fp16, reduce=fp32, buffer=fp16)`
policy, which works on Pascal (no BF16). When mixing GPU generations on a single host,
NCCL's default P2P/IB transports often fail to negotiate — set the following env vars
before launching `torchrun`:

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO        # optional, helpful for first run
torchrun --nproc_per_node=2 train.py
```

`torch.compile` is **off by default** in this setup. It must produce identical module
graphs on every rank or FSDP's flat-param layout diverges; pass `--compile` only when
all GPUs have compute capability >= 7.0 (the script enforces this with an all-reduce
and silently disables compile if any rank fails the check).

The slower GPU dictates step time — expect the 1070 Ti to bottleneck a 3070 Ti pair.
This is intentional for the experiment: the goal is to study sharded-comm patterns,
not maximize throughput.

### Eval comparability
`experiment.sh` always evaluates on the same shard (`chunk_0049.bin`) with the same fixed settings (50 batches, seq_len 1024), so **all runs in `results.md` are directly comparable**. Do not override `--seq_len` or `--max_batches` when running through `experiment.sh` — it would break comparability. If you need a quick sanity check with different settings, run `eval_checkpoint.py` directly in the terminal instead.
