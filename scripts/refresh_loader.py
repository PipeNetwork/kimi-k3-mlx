#!/usr/bin/env python3
"""Refresh the bundled `kimi_k3.py` in every published Kimi-K3 MLX build.

Published repos ship the loader alongside the weights, so a fix to the loader
does not reach anyone until each repo's copy is replaced. The KDA gate fix
(87f4cf4) is one of those: the weights in those repos are fine, the bundled
kimi_k3.py is not.

Compares by content hash and skips repos already current, so it is safe to
re-run and it reports exactly what it changed.

  refresh_loader.py            # dry run: list what would change
  refresh_loader.py --push     # actually upload
"""
import argparse
import hashlib
import os
import sys

from huggingface_hub import HfApi, hf_hub_download

OWNER = "pipenetwork"
LOCAL = "/Users/david/llm/kimi-k3-mlx/kimi_k3.py"
TARGET = "kimi_k3.py"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="upload; otherwise dry run")
    ap.add_argument("--filter", default="Kimi-K3",
                    help="only consider repos whose id contains this")
    a = ap.parse_args()

    api = HfApi()
    want = open(LOCAL, "rb").read()
    want_h = sha(want)
    print(f"local {TARGET}: {len(want)} bytes, sha {want_h}\n")

    repos = [m.id for m in api.list_models(author=OWNER) if a.filter in m.id]
    print(f"{len(repos)} repos matching {a.filter!r}\n")

    stale, current, absent, failed = [], [], [], []
    for rid in sorted(repos):
        try:
            files = {s.rfilename for s in api.model_info(rid).siblings}
        except Exception as e:
            failed.append((rid, f"info: {e}")); continue
        if TARGET not in files:
            # Not every K3 repo bundles the loader (calibration/plan repos do
            # not). Adding it where it never existed is not a refresh.
            absent.append(rid); continue
        try:
            p = hf_hub_download(rid, TARGET)
            got_h = sha(open(p, "rb").read())
        except Exception as e:
            failed.append((rid, f"download: {e}")); continue
        (current if got_h == want_h else stale).append((rid, got_h))

    for rid, h in stale:
        print(f"  STALE   {rid}  ({h})")
    print(f"\nstale {len(stale)} | current {len(current)} | "
          f"no {TARGET} {len(absent)} | errors {len(failed)}")
    for rid, why in failed:
        print(f"  ERROR   {rid}: {why}")
    if absent:
        print("  (no loader bundled: " + ", ".join(absent[:6])
              + (" ..." if len(absent) > 6 else "") + ")")

    if not a.push:
        print("\ndry run -- pass --push to upload")
        return
    if not stale:
        print("\nnothing to do")
        return

    print()
    ok = 0
    for rid, _ in stale:
        try:
            api.upload_file(path_or_fileobj=LOCAL, path_in_repo=TARGET,
                            repo_id=rid, repo_type="model",
                            commit_message="Fix KDA gate: A_log is per-head; "
                                           "safe-gate is sigmoid, not a softplus clamp")
            ok += 1
            print(f"  pushed  {rid}")
        except Exception as e:
            print(f"  FAILED  {rid}: {e}")
    print(f"\n{ok}/{len(stale)} updated")
    return 0 if ok == len(stale) else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
