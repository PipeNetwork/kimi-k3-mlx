# Two-node unpruned inference

This deployment targets two Mac Studios with M3 Ultra and 512 GiB unified
memory each. Production inference requires a direct Thunderbolt 5 link, macOS
RDMA, and the JACCL backend in MLX. Ethernet remains useful for SSH and model
downloads but is not an accepted inference transport.

## Proven baseline and checkpoint

The primary checkpoint is
[`kernelpool/Kimi-K3-2bit-UVMAX`](https://huggingface.co/kernelpool/Kimi-K3-2bit-UVMAX)
at immutable revision
`edb5113218df612f4a92f95145680f3f8eacd375`. Its index reports
2,779,483,539,072 parameters and 816,773,159,296 weight bytes (760.68 GiB).
Routed expert FFNs—93.8% of parameters—are affine 2-bit/group-128. Attention is
6-bit, shared/latent MoE and routers are 8-bit, embeddings and LM head are
4-bit, and normalization/AttnRes state remains BF16.

Tarjei Mandt's
[`mlx-lm` PR #1626](https://github.com/ml-explore/mlx-lm/pull/1626) demonstrates
this exact checkpoint over JACCL on two M3 Ultras. This fork pins that loader as
the proven compatibility base, adds stable-release compatibility, and uses a
pipeline-local disk layout because neither internal 1 TB SSD can hold a full
checkpoint plus safe working space.

## Partition and storage

MLX pipelines execute from the highest rank toward rank zero. Kimi-K3 has 93
decoder layers, of which only layer 0 is dense. The gap-free split is:

| Host | Rank | Decoder layers | MoE layers | Local weight files |
|---|---:|---:|---:|---:|
| Beast1 | 0 | `[47, 93)` | 46 | 384.301 GiB |
| Beast2 | 1 | `[0, 47)` | 46 plus dense layer 0 | 384.462 GiB |

The embedding, final norm/AttnRes, and LM head endpoint shards are duplicated
because released mlx-lm generation samples on every rank. Vision weights are
not downloaded for the text benchmark. The boundary sends one packed BF16
message containing the current hidden state and accumulated AttnRes blocks;
the receiver reconstructs inverse-RMS state locally. Boundary transfer and
final hidden-state synchronization use two independent JACCL backend instances,
both on the same mandatory RDMA fabric. After one initialization collective on
each mesh, the payload mesh is point-to-point only: rank 1 fully materializes
and eagerly sends the boundary packet, while rank 0 receives it and computes
all 46 local MoE layers before any further communication. Rank 1 cannot enter
the final synchronization until its send has retired. The second mesh then
distributes rank 0's completed, row-contiguous hidden state with an eager
all-gather. Keeping the one-way payload transfer and the final collective on
different RDMA queue pairs avoids both the sender-retirement deadlock and the
full-model `memmove` faults observed when the boundary itself was an
all-gather. The stress test transfers the
production-shaped 5 x 1 x 64 x 7168 BF16 packet before checking bit-exact
miniature-model prefill and cached decode parity.

The original source-to-2-bit route remains available through
`scripts/convert.py` and `scripts/build_pipeline_stage.sh`. It downloads source
shards on demand, commits each converted layer transactionally, evicts source
weights after use, and resumes from its journal. The published UVMAX checkpoint
is preferred for first deployment because it saves conversion time and keeps
quality-sensitive tensors at higher precision.

## Software

Use only the lockfile environment:

```bash
uv sync --frozen
scripts/install_model.sh .venv/bin/python
```

The key versions are MLX 0.32.0 and mlx-lm 0.31.3. MLX 0.32.0 includes JACCL
barrier, race, and multi-wire receive fixes. Kimi-K3 support is carried locally
from the pinned upstream contribution until it lands in a stable mlx-lm
release; this is recorded in `kimi_k3_uvmax.py` and `NOTICE.md`.

## Stage download

Run both downloads concurrently. They are revision-pinned, size-validated,
resumable, and use Hugging Face Xet high-performance mode on internal NVMe.

```bash
# Beast1
.venv/bin/python scripts/prepare_uvmax_stage.py --rank 0 --verify-sha256

# Beast2
ssh beast2.local 'cd ~/dev/kimi-k3-mlx-distributed && \
  .venv/bin/python scripts/prepare_uvmax_stage.py --rank 1 --verify-sha256'
```

Each `stage-manifest.json` records the source commit, exact sizes, rank, layer
interval, and proof that every completed shard matched the SHA-256 supplied by
the Hub. The production runner rejects an incomplete, truncated, or unverified
stage.

## Thunderbolt RDMA and JACCL

1. Connect a certified Thunderbolt 5 cable directly between the two Studios.
2. Keep Ethernet available for SSH; it is not selected for inference.
3. Verify at least one `PORT_ACTIVE` on both machines:

   ```bash
   /usr/bin/ibv_devinfo
   ssh beast2.local /usr/bin/ibv_devinfo
   ```

4. Generate the machine-specific JACCL hostfile:

   ```bash
   scripts/configure_jaccl.sh
   ```

5. Run inference:

   ```bash
   scripts/run_distributed.sh \
     --prompt 'Who is Albert Einstein?' \
     --max-tokens 256
   ```

`scripts/distributed_generate.py` explicitly calls
`mx.distributed.init(strict=True, backend="jaccl")` and requires group size 2.
No code path silently selects ring, MPI, Ethernet, or a single process.

## Validation order

Before reporting a result:

1. `scripts/test_all.sh` passes in the frozen environment.
2. The two-rank worker passes the production-sized boundary transfer plus
   64-token prefill and 32 cached decode steps, first locally and then on real
   JACCL/RDMA.
3. Both stage manifests are complete and all byte sizes validate.
4. RDMA reports active on both machines and JACCL initialization succeeds.
5. A deterministic 32-token smoke prompt produces coherent output.
6. Three benchmark prompts run at 256 generated tokens under the methodology in
   [BENCHMARKING.md](BENCHMARKING.md).

Do not call a TCP-ring development check an RDMA result, and do not publish a
number from a hand-written decode loop that did not wire model memory.
