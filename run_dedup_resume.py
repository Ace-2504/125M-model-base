"""Resumable Phase-2 writer stage (stage 3 only) — memory- and network-safe.

Stages 1 (MinHash sigs) and 2 (LSH near_dups.json) already completed and are on
disk; only the corpus-writer stage crashed (each of 15 legal writers independently
rebuilt the CaseHOLD/LexGLUE contamination n-gram set -> memory blowup at 4 workers
on a 16 GB box, and repeated network streaming was flaky).

This driver builds the contamination set ONCE (with retries), caches it, and runs
the writers at low concurrency with the cached set injected. It reuses the verbatim
`pipeline.write_corpus_shard`; it does not modify any of the four canonical files.

MUST be launched with PYTHONHASHSEED=0 so the integer word-ngram hashes are identical
across processes (otherwise the cached contam hashes would never match doc hashes and
decontamination would silently do nothing).

  PYTHONHASHSEED=0 python run_dedup_resume.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import sys
import time

import config
import pipeline

CONTAM_CACHE = os.path.join(config.DATA_ROOT, "tmp", "contam_ngrams.pkl")


def build_contam_cached() -> str:
    if os.path.exists(CONTAM_CACHE):
        print(f"contam cache exists: {CONTAM_CACHE}")
        return CONTAM_CACHE
    last = None
    for attempt in range(1, 7):
        try:
            print(f"building contamination n-grams (attempt {attempt}/6)...")
            grams = pipeline._build_contamination_ngrams()
            if not grams:
                raise RuntimeError("empty contamination set")
            with open(CONTAM_CACHE, "wb") as fh:
                pickle.dump(grams, fh, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"cached {len(grams):,} contamination n-grams -> {CONTAM_CACHE}")
            return CONTAM_CACHE
        except Exception as e:
            last = e
            print(f"  !! {type(e).__name__}: {e}; retrying in {5*attempt}s")
            time.sleep(5 * attempt)
    raise RuntimeError("failed to build contamination set") from last


_CONTAM = None


def _load_contam():
    global _CONTAM
    if _CONTAM is None:
        with open(CONTAM_CACHE, "rb") as fh:
            _CONTAM = pickle.load(fh)
    return _CONTAM


def _writer(args):
    source_name, shard_basename = args
    # Inject the cached contam set so write_corpus_shard doesn't rebuild it.
    pipeline._build_contamination_ngrams = _load_contam
    return pipeline.write_corpus_shard(source_name, shard_basename)


def main() -> None:
    assert os.environ.get("PYTHONHASHSEED") == "0", "launch with PYTHONHASHSEED=0"
    build_contam_cached()

    work = [(src, f"shard-{i:03d}.txt")
            for src, n in pipeline.CLEAN_SHARDS.items() for i in range(n)]
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print(f"writing {len(work)} corpus shards across {workers} processes...")
    with mp.Pool(workers) as pool:
        results = pool.map(_writer, work)
    pipeline.write_phase2_report(results)


if __name__ == "__main__":
    main()
