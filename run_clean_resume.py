"""Resumable, retry-wrapped driver for Phase 1 clean shards.

Does NOT modify the four canonical files. It imports the verbatim
`clean_shard` from pipeline.py and re-runs only the shards that did not
cleanly finish (transient HuggingFace stream drops abort pool.map wholesale).
Runs sequentially with retries to be gentle on a flaky connection / limited RAM.

Usage:
  python run_clean_resume.py case-law:0,1 sec:2,3 fineweb-edu:3,4
"""

from __future__ import annotations

import sys
import time

import config
import pipeline


def _urls_for(source) -> list[str]:
    cfg = source.config_name or "default"
    urls = pipeline._parquet_urls(source.hf_id, cfg, source.split)
    if source.name == "fineweb-edu":
        urls = urls[:5]  # matches phase_clean(fineweb_shards=5)
    return urls


def run_shard(source, shard_index: int, urls: list[str], attempts: int = 6) -> dict:
    per_shard_cap = source.token_budget // max(1, len(urls))
    url = urls[shard_index]
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"--- {source.name} shard {shard_index:03d} (attempt {attempt}/{attempts}) ---")
            return pipeline.clean_shard(source.name, url, shard_index, per_shard_cap)
        except Exception as e:  # transient HTTP/stream errors mostly
            last_exc = e
            wait = min(60, 5 * attempt)
            print(f"    !! {type(e).__name__}: {e}\n    retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"{source.name} shard {shard_index} failed after {attempts} attempts") from last_exc


def main() -> None:
    by_name = {s.name: s for s in config.DATA_MIX}
    targets: list[tuple] = []
    for arg in sys.argv[1:]:
        name, _, idxs = arg.partition(":")
        source = by_name[name]
        for i in idxs.split(","):
            targets.append((source, int(i)))

    results = []
    for source, i in targets:
        urls = _urls_for(source)
        results.append(run_shard(source, i, urls))

    print("\nRESUME DONE")
    for r in results:
        print(f"  {r['source']:<12} shard {r['shard']:03d} kept={r['kept']} "
              f"est_tokens={r['est_tokens']/1e6:.1f}M reasons={r['reasons']}")


if __name__ == "__main__":
    main()
