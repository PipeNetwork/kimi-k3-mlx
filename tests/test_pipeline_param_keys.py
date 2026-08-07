#!/usr/bin/env python3
"""Mimic sharded_load()'s shard-selection walk on a pipelined model.

sharded_load picks which safetensors files to download by walking
tree_flatten(model.parameters()) after pipeline() and looking each key up in the
weight index. Two ways that goes wrong and neither shows up in a forward pass:

  * a non-array attribute stashed on the module (pipeline_group) leaking into
    parameters(), producing a key with no index entry -> ValueError at load;
  * parameters() reporting layers this rank does not own, which would make every
    rank download the whole 451 GB.

Run under mlx.launch with >=2 ranks.
"""
import os
import sys

import mlx.core as mx
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_kimi_k3 import kimi_k3, tiny_args  # noqa: E402


def main():
    group = mx.distributed.init()
    rank, size = group.rank(), group.size()

    mx.random.seed(0)
    model = kimi_k3.Model(tiny_args())
    n_all = len({k.split(".layers.")[1].split(".")[0]
                 for k, _ in tree_flatten(model.parameters()) if ".layers." in k})

    model.model.pipeline(group)
    keys = [k for k, _ in tree_flatten(model.parameters())]
    vals = [v for _, v in tree_flatten(model.parameters())]

    lo, hi = model.model.start_idx, model.model.end_idx
    seen = sorted({int(k.split(".layers.")[1].split(".")[0])
                   for k in keys if ".layers." in k})
    shared = sorted({k for k in keys if ".layers." not in k})
    non_arrays = [k for k, v in zip(keys, vals) if not isinstance(v, mx.array)]

    ok = True
    if non_arrays:
        print(f"rank {rank}: NON-ARRAY leaked into parameters(): {non_arrays}")
        ok = False
    if seen != list(range(lo, hi)):
        print(f"rank {rank}: layer mismatch, owns [{lo}:{hi}] but parameters() "
              f"reports {seen}")
        ok = False
    if any("pipeline_group" in k for k in keys):
        print(f"rank {rank}: pipeline_group present in parameters()")
        ok = False

    print(f"rank {rank}/{size}: layers[{lo}:{hi}] -> {len(keys)} param keys, "
          f"layers seen {seen[:3]}..{seen[-1:]}, {len(shared)} shared, "
          f"{'OK' if ok else 'BAD'}", flush=True)
    print(f"rank {rank}: shared keys = {shared}", flush=True)

    total = mx.distributed.all_sum(mx.array([float(len(seen))]), group=group)
    mx.eval(total)
    if rank == 0:
        covered = int(total.item())
        print(f"\nlayers covered across ranks: {covered} of {n_all}")
        good = ok and covered == n_all
        print("PARAM-KEY CHECK " + ("OK" if good else "FAIL"))
        sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
