"""
Download FineWeb-Edu (sample-10BT by default), tokenize with the Mistral
tokenizer, and shard into uint16 chunk_XXXX.bin files that are format-compatible
with BinDataset in train.py and eval_checkpoint.py.

Examples
--------
# Quick laptop run (~3 shards ≈ 300M tokens ≈ 600MB on disk)
python prepare_data.py --out_dir data --n_shards 3

# Full sample-10BT equivalent (~50 shards ≈ 10B tokens ≈ 20GB)
python prepare_data.py --out_dir data --n_shards 50

# Swap tokenizer (any 32k-vocab tokenizer works; must be <= 65535 for uint16)
python prepare_data.py --tokenizer NousResearch/Llama-2-7b-hf
"""
import os
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count


TOKENIZER = None  # populated in each worker via _init_worker


def _init_worker(tokenizer_name: str):
    global TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)


def tokenize_doc(args):
    text, eot_id = args
    ids = TOKENIZER.encode(text, add_special_tokens=False)
    ids.append(eot_id)
    arr = np.asarray(ids, dtype=np.uint32)
    assert arr.max() < 2**16, f"token id {arr.max()} overflows uint16"
    return arr.astype(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",          default="data")
    ap.add_argument("--tokenizer",        default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--dataset",          default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--dataset_config",   default="sample-10BT")
    ap.add_argument("--split",            default="train")
    ap.add_argument("--n_shards",         type=int, default=50)
    ap.add_argument("--tokens_per_shard", type=int, default=100_000_000)
    ap.add_argument("--n_proc",           type=int, default=max(1, cpu_count() // 2))
    ap.add_argument("--overwrite",        action="store_true",
                    help="re-tokenize shards that already exist on disk")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eot_id = tok.eos_token_id
    assert eot_id is not None, f"tokenizer {args.tokenizer} has no eos token"
    print(f"[tok]  {args.tokenizer}  vocab={tok.vocab_size}  eos={eot_id}")
    print(f"[data] {args.dataset} :: {args.dataset_config} (streaming)")
    print(f"[out]  {args.n_shards} shard(s) × {args.tokens_per_shard:,} tokens "
          f"→ {args.out_dir}/  ({args.n_proc} workers)")

    def shard_path(i: int) -> str:
        return os.path.join(args.out_dir, f"chunk_{i:04d}.bin")

    shard_idx = 0
    if not args.overwrite:
        while shard_idx < args.n_shards and os.path.exists(shard_path(shard_idx)):
            print(f"[skip] {shard_path(shard_idx)} exists")
            shard_idx += 1
        if shard_idx >= args.n_shards:
            print("[done] all requested shards already present; pass --overwrite to redo")
            return

    ds = load_dataset(
        args.dataset,
        name=args.dataset_config,
        split=args.split,
        streaming=True,
    )

    shard_buf = np.empty(args.tokens_per_shard, dtype=np.uint16)
    shard_pos = 0
    pbar = tqdm(total=args.tokens_per_shard, desc=f"shard {shard_idx:04d}",
                unit="tok", unit_scale=True)

    with Pool(args.n_proc, initializer=_init_worker, initargs=(args.tokenizer,)) as pool:
        doc_stream = ((doc["text"], eot_id) for doc in ds)
        for tokens in pool.imap(tokenize_doc, doc_stream, chunksize=16):
            n = len(tokens)
            offset = 0
            while offset < n:
                room = args.tokens_per_shard - shard_pos
                take = min(room, n - offset)
                shard_buf[shard_pos:shard_pos + take] = tokens[offset:offset + take]
                shard_pos += take
                offset    += take
                pbar.update(take)

                if shard_pos == args.tokens_per_shard:
                    path = shard_path(shard_idx)
                    shard_buf.tofile(path)
                    pbar.close()
                    print(f"[shard] wrote {path}  ({args.tokens_per_shard:,} tokens)")

                    shard_idx += 1
                    shard_pos  = 0
                    if shard_idx >= args.n_shards:
                        return
                    pbar = tqdm(total=args.tokens_per_shard,
                                desc=f"shard {shard_idx:04d}",
                                unit="tok", unit_scale=True)

        if shard_pos > 0 and shard_idx < args.n_shards:
            path = shard_path(shard_idx)
            shard_buf[:shard_pos].tofile(path)
            pbar.close()
            print(f"[shard] wrote {path}  ({shard_pos:,} tokens, partial)")


if __name__ == "__main__":
    main()
