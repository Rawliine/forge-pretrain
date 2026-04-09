"""Evaluate a checkpoint on held-out data shards."""
import argparse, glob, math, numpy as np, torch
from model import get_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoint.pt")
    p.add_argument("--data", default="/home/data/chunk_0049.bin",
                   help="Comma-separated shard paths or glob pattern")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--max_batches", type=int, default=50)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=True, map_location="cuda")
    model = get_model(ckpt["config"]).cuda()
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params = {n_params}")
    print(f"params_m = {n_params/1e6:.1f}M")

    shards = []
    for s in args.data.split(","):
        shards.extend(glob.glob(s.strip()))
    assert shards, f"No shards found: {args.data}"

    total_loss, count = 0.0, 0
    with torch.no_grad():
        for shard in shards:
            data = torch.from_numpy(
                np.fromfile(shard, dtype=np.uint16).astype(np.int64)
            )
            for i in range(0, len(data) - args.seq_len - 1, args.seq_len):
                if count >= args.max_batches:
                    break
                x = data[i : i + args.seq_len].unsqueeze(0).cuda()
                y = data[i + 1 : i + 1 + args.seq_len].unsqueeze(0).cuda()
                _, loss = model(x, y)
                total_loss += loss.item()
                count += 1
            if count >= args.max_batches:
                break

    avg = total_loss / max(count, 1)
    ppl = math.exp(avg)
    print(f"val_loss = {avg:.4f}")
    print(f"perplexity = {ppl:.2f}")
    print(f"batches = {count}")

if __name__ == "__main__":
    main()
