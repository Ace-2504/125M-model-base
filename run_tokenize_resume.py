"""Memory-safe Phase-4 tokenize driver.

pipeline.tokenize_shard encodes ENCODE_BATCH=1000 docs at once; SEC docs are
huge (~24k tokens each), so one batch balloons to ~1 GB of Python token lists
per worker and 4 concurrent workers OOM a 16 GB box.

Reducing ENCODE_BATCH is OUTPUT-PRESERVING: tokens flow through a persistent
`buf`, so window boundaries and the every-100th-window val split are identical
regardless of batch size — only flush frequency changes. This driver monkeypatches
ENCODE_BATCH down and runs the verbatim tokenize_shard at low concurrency. It does
not modify any of the four canonical files.

  python run_tokenize_resume.py [workers] [batch]
"""

from __future__ import annotations

import multiprocessing as mp
import sys

import pipeline

BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 100


def _worker(args):
    pipeline.ENCODE_BATCH = BATCH  # shrink batch inside the child process
    return pipeline.tokenize_shard(*args)


def main() -> None:
    work = [(name, i, n) for name, n in pipeline.TOKENIZE_SHARDS.items() for i in range(n)]
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"tokenizing {len(work)} shards across {workers} processes (ENCODE_BATCH={BATCH})...")
    with mp.Pool(workers) as pool:
        results = pool.map(_worker, work)
    pipeline.write_token_index(results)


if __name__ == "__main__":
    main()
