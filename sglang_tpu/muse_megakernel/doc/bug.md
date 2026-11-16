# Bug: column-sliced pool / unpool (SP + EP decode) changes numerics

Status: resolved. It was a harness artifact, not a kernel bug (see Root cause). The debug tokens and SGLANG_TPU_MUSE_MK_EP_COL_POOL_PARTS have been removed; the 3-layer harness now lives at `muse-mk-tools/q8b1/muse_spark_decode_spep_l3.py`. Code is uncommitted in the working tree of `junyanxu/muse-mk-layout`
(`decode/spep/{layer,config}.py`, `sglang_tpu/envs.py`); `git diff` shows it.

## What the feature does

Decode SP + EP a2a arm (`decode/spep/`). Each rank owns `rc = rows / tp` rows. Today the pool
(`[H, He]` fp8) and unpool (`[He, H]`) are replicated: each rank reads the whole 32 MiB matrix
for its own rows (`_pool`, `_unpool_rows` in `layer.py`). Column slicing (`_pool_cols`,
`_unpool_cols`) reads only this rank's `1/tp` of the columns instead:

1. gather every rank's input rows: `_bcast_own_rows` writes them to HBM `g`, then
   `_hbm_rows_to_lhs` copies them into `S["lhs"]`;
2. `stream_matmul` over this rank's column block, for every row;
3. return each owner its rows of that block: `_stage_by_owner`, then `_blocks_to_owners`;
4. `_own_rows_of_blocks` concatenates the blocks back into full-width own rows.

The output should be bit-identical to the replicated path, because each output column is the
same dot product. It is not.

## Knobs

| env | meaning |
|---|---|
| `SGLANG_TPU_MUSE_MK_EP_COL_POOL=1` | turn the feature on (also allocates `col_pool_scratch`) |
| `SGLANG_TPU_MUSE_MK_EP_COL_POOL_PARTS` | default `pool,unpool`; debug tokens below |

Debug tokens (temporary, in `_pool_cols` / `_pool`):

| token | effect |
|---|---|
| `fullmm` | compute the full replicated matmul on every row into `ep_u`, then take this rank's block |
| `local` | use own rows straight from `ep_u`; skip the block return |
| `nogather` | skip step 1 |
| `xchg` | run `_blocks_to_owners` but discard its result |
| `selfonly` / `peersonly` | take only this rank's block / only peers' blocks from `ep_prcv`, the rest from `ep_u` |
| `selftest_g` / `selftest_x` | in the replicated `_pool`: run the gather / the block exchange on synthetic integer data, and add 1 to x on any mismatch |

## Repro

Run anvil with `tpu-perf-forge/anvil/examples/muse_spark_decode_spep_l3.py`, which is the 62-layer
harness `muse_spark_decode_spep.py` cut to 3 layers. Both scripts are uncommitted. The harness
reads `N_LAYERS` at import time, so it cannot be set through `--env`.

```bash
cd /home/junyanxu_google_com/nfs/code/tpu-perf-forge
python3 -m anvil --controller 172.27.0.4:50060 sweep \
  --script /home/junyanxu_google_com/nfs/code/muse-mk-tools/q8b1/muse_spark_decode_spep_l3.py \
  --checkout /home/junyanxu_google_com/nfs/code/sglang-muse-disagg-kernel \
  --dim mode=2 --dim q=8 --dim batch=2 --dim context=32768 \
  --env SGLANG_TPU_MUSE_MK_EP_COL_POOL_MAX_ROWS=16 \
  --control-env SGLANG_TPU_MUSE_MK_EP_COL_POOL_MAX_ROWS=0 --detach
```

(Today's knob; the bisection below ran SGLANG_TPU_MUSE_MK_EP_COL_POOL=1 with the PARTS tokens, since removed.) Read the result with `curl http://172.27.0.4:50070/v1/sweeps/<id>`, under
`per_shape[].diffs[].vs_control`. That value is the worst row's `max|a - b| / max|b|`; 0 means
bit-identical. Geometry: tp = 8 (2x2x1, 4 chips x 2 cores), 16 rows at b2 (`rc = 2`, window 16),
H 8192, He 4096, column block W = He / 8 = 512.

## Bisection (3 layers, q8 b2 unless noted)

| sweep | parts | diff out0 / out1 |
|---|---|---|
| `20261116-025403-3751` | pool,unpool (b1 and b2) | 1.775 / 1.457 (same at b1 and b2: deterministic) |
| `20261116-025821-6dca` | pool | 1.918 / 1.652 |
| `20261116-025829-5766` | unpool | 1.793 / 1.727 |
| `20261116-030108-0629` | none (scratch only) | 0 / 0 |
| `20261116-030422-d17e` | pool,fullmm | 1.918 / 1.652 (the column matmul is not the cause) |
| `20261116-031052-5f8e` | pool,fullmm,local | 0 / 0 |
| `20261116-031100-e0e9` | pool,fullmm,local,nogather | 0 / 0 |
| `20261116-030903-57f5` | pool, after the aligned-slot rewrite | 1.918 / 1.652 (unchanged) |
| `20261116-031256-a4f4` | pool,fullmm,local,xchg | 0 / 0 (the exchange clobbers nothing) |
| `20261116-031441-4c8f` | ...,xchg,selfonly | 0 / 0 |
| `20261116-031449-1bcc` | ...,xchg,peersonly | 1.918 / 1.652 |
| `20261116-032507-9d3d` | selftest_g | 0 / 0 (the gather is correct on synthetic rows) |
| `20261116-032515-6a82` | selftest_x | 0 / 0 (the block exchange is correct on synthetic blocks) |
| `20261116-032729-eb1f` / `-032737-ae26` | pool / pool,unpool, harness fixed (below), b1 and b2 | 0.0078 / 0 (out0: one bf16 ulp, 2^-7, in one row; from the rms sum order over 8 concatenated 512-column blocks vs one 4096-column block) |

What this shows:

- This rank's own block is right. Its own rows survive the gather, and the local DMA in
  `_blocks_to_owners` delivers them.
- Peers' blocks of my rows are wrong. A peer computes them from its gathered copy of my input
  rows. So the fault is in one of two places:
  - (a) the gather (`_bcast_own_rows` + `_hbm_rows_to_lhs`) delivers other ranks' rows wrong.
    `local` only ever checked a rank's own rows.
  - (b) the remote copies in `_blocks_to_owners`, or the concatenation, are wrong.
- The selftests tell (a) from (b).

## Root cause: the harness weights are not replicated

The harness fills every weight with `_fill` over the global `P("model")`-sharded shape, so each
rank's "replicated" `wpool` / `wunpool`, its `smalls` (the pool scale rows) and its `emb` rows
hold different random values. The replicated path never mixes ranks' copies. The column-sliced
path computes my rows' block d on rank d, with rank d's weights, which explains why only the
peers' blocks are wrong. The harness now tiles one rank's block across ranks for `smalls`,
`wpool`, `wunpool` and `emb` (`REPLICATED`, `_tiled`). With that fix the column-sliced path matches the replicated one to one bf16 ulp. Absolute timings from before the fix used different weights; re-measure both arms rather than mixing old and new numbers.

## Why (a) mattered beyond this feature

The baseline SP + EP decode uses the same gather for every layer's pre-attention input:
`_gather_lhs`, then `_pre_attn_lhs` -> `_hbm_rows_to_lhs`. Attention runs every row on every
rank (heads split), and the o-proj row reduce-scatter adds each rank's partials. If peers'
copies of a row are wrong, the baseline SP + EP output is wrong too. An A/B against itself
cannot show that, so it has never been checked. Check it against TP (mode 0) on matching
weights, or against a host reference.

## Things already ruled out

- Uninitialized memory or a scratch layout shift: scratch only gives diff 0.
- The column-sliced matmul (`n_off`, `tn`, `_scale_slice`): `fullmm` gives the same diff.
- Row DMAs at offsets off the (8, 128) tile in `_blocks_to_owners`. It now sends one aligned
  `[rows_own_padded, W/2]` slot per owner (`_stage_by_owner`); the diff did not change. Keep
  this rewrite anyway.
- A semaphore race: every diff is identical across b1 / b2 and across reruns.

## Code pointers (`sglang_tpu/muse_megakernel/decode/spep/layer.py`)

- `_bcast_own_rows`: on tp = 8 it sends to the partner core and to the same core on each other
  chip; the partner relays the other chips' rows on-chip (`lhs_relay`). Rows have
  `rc % 8 != 0`, so it goes row by row: the source is `own[off + r]` (traced `off = w_off`), the
  destination is HBM `g[k * rc + r]`.
- `_hbm_rows_to_lhs`, `_stage_by_owner`, `_blocks_to_owners`, `_own_rows_of_blocks`,
  `_pool_cols`, `_unpool_cols`.
- `_gather_lhs` (baseline) calls `_bcast_own_rows` with `off = 0` on the same `ep_lhs_g` /
  `ep_lhs_sems` / `ep_lhs_rsems`.
- `_unpool_cols` gathers into `ep_ug` but shares `ep_lhs_sems` / `ep_lhs_rsems` with
  `_gather_lhs`, whose rows are a different width (`He/2` vs `H/2` words). That is a possible
  semaphore-aliasing hazard; not seen yet.

## Cleanup when fixed

Remove the debug tokens and `SGLANG_TPU_MUSE_MK_EP_COL_POOL_PARTS`, and delete
`muse_spark_decode_spep_l3.py`. Then run the 62-layer A/B at b1 / b2 / b5 / b8 (perf: xprof
device time; diff must be 0) and redraw `decode_q8b2_tp_vs_spep.svg` with
`muse-mk-tools/q8b1/chart.sh`.
