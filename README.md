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
