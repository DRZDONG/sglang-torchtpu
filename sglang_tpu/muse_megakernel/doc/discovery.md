# Muse megakernel: discoveries

Each entry is a finding: what it is, the evidence (anvil sweep ids, all on the 2x2x1 single host, xprof device time), and why it holds. Negative results state why they failed.

## Measurement

- **The anvil harness filled replicated weights per rank.** `_fill` hashed the global `P("model")` shape, so each rank's `wpool`, `wunpool`, `smalls` and `emb` differed. Only paths that mix ranks' copies see it: column-sliced pool / unpool computed a row's block d on rank d with rank d's weights. The fix tiles one rank's block (`REPLICATED`, `_tiled` in `muse_spark_decode_spep.py`). Numbers measured before the fix are not comparable with numbers after it. The full bisection is in `bug.md`.
- **A 62-layer A/B diff is not evidence once numerics change at all.** One bf16 ulp in one row at 3 layers (0.0078) became a worst-row error of 1.7-2.2 at 62 layers, because routing top-k flips propagate. Check numerics on a 3-layer copy of the harness (`muse-mk-tools/q8b1/muse_spark_decode_spep_l3.py`). The harness reads `N_LAYERS` at import, so `--env` cannot set it.
- **Pinning `--runner` serializes a sweep.** Without it a sweep fans out over every READY runner.
- **Deep drills dump trace and LLO from one build**: `--keep-trace all --iters 1 --llo all`.

## Decode (q8, 32k unless noted)

- **EP gather beats SP+EP a2a up to 128 rows.** b1 8.85 vs 11.01 ms; b2 11.87 vs 14.30 (TP 6.32 / 10.00), from `20261116-033833-d497`; at every q, context and batch in `20261116-034534-39b0`, `-040412-7631`, `-034542-7b82`. Both hold the same weights. At 16 rows each collective round costs a few us of latency whatever its bytes, and the a2a arm takes about 9 rounds per layer against the gather arm's 4-5. Its extra rounds:
  - the o row reduce-scatter in 8-row blocks;
  - a row gather plus a block return around the pool, and again around the unpool;
  - the count exchange before the ragged dispatch;
  - the next layer's row gather.
- **The gather arm's gap to TP is mostly expert imbalance.** At b2, region trace `20261116-034120-2400`, gather is 2.47 ms/step over TP:
  - waiting in u_rs for the busiest rank: +1.65 ms. That rank holds 15.5 experts against a mean of 11.2, and its experts take 8.34 vs 6.30 ms/step. TP splits every expert evenly.
  - o RS + AG: +0.44 ms.
  - the f AG wait: +0.31 ms.
  - the u RS itself: +0.28 ms.
- **The gather arm stops at 128 rows.** Its replicated per-row VMEM buffers total about 110-125 KiB/row (o RS/AG 36, pool 16, u RS/AG and unpool 56). At 136 rows the program needs 62-90 MiB of the ~64 MiB VMEM.
- **The expert loop is not DMA-bound.** With the arms skipped (`SGLANG_TPU_EP_DEV_SKIP=experts`, `20261116-025017-e1de`), a 3 MiB chunk's four DMAs take 0.91 us (3.5 TB/s), under the q8 arm's 0.97 us. The q8 arm is the same in TP's and EP's LLO (768 fp8 pushes, 96 matmuls, 1536 s4 unpacks per chunk, `20261116-020301-54f1`). EP spends about 0.2 us/chunk around it (TP about 0.1): all four weight DMAs are awaited before the arm, and the item zero / readout runs at 32 rows.
- **Column-sliced pool / unpool on the a2a arm pays only at 16 rows.** Change per step: -0.4 ms at 16 rows; +0.27 / +0.55 / +0.35 / +0.98 ms at 24 / 32 / 40 / 64 rows (`20261116-032949-7e2f`, `-033557-98c7`). Gathering every row's full input (R x H bf16, about 1 MiB per rank per layer at 64 rows) costs more ICI time than the 56 MiB of weight reads saves. Hence `SGLANG_TPU_MUSE_MK_EP_COL_POOL_MAX_ROWS=16`.
- **EP decode MoE tiles.** An item is at most 32 rows of one expert, run on the q8 / q16 / q32 arm. A busier expert splits into items, and each item re-streams the expert's weights. Weights stream in I-chunks of 512 (3 MiB) through the expert ring. w13 runs one 256-column block at a time; w2 runs 512-column tiles. K = 4096 goes in 16 blocks of 256, each partial scaled by its block scale. At R <= 32 that scaling is cheap.

### Tried on the a2a arm, and why they failed

- **One o RS DMA per owner, after the last column tile: +0.2 ms at every batch** (`20261116-023458-2c10`). The bytes are unchanged: each rank owns 2 rows, but a DMA moves an 8-row tile, so about 896 KiB per rank per layer are sent against about 112 KiB of live data. The original per-tile sends overlapped the o-proj matmul; sending once at the end loses that overlap.
- **o through TP's column RS + AG: -0.14 ms at b1, +0.74 ms at b8** (`20261116-024041-2689`). The column blocks are dense, but the path needs `fold_o` off and a post-attention pass over every row. At 64 rows the row RS had no padding left to save.
- **Waiting for w2 between gate|up and down: +0.9 ms at b2** (`20261116-024653-189a`). A DMA wait inside the switch arm splits the arm's straight-line code, and the scheduler no longer overlaps w2's MXU work with gate|up.

## Prefill SP + EP (one 4096-row chunk)

- **Where the chunk goes** (trace + LLO `20261116-043851-2a74` at ctx 0, `20261116-043858-d096` at 756k):

  | | ctx 0 (158 ms) | 756k (767 ms) |
  |---|---|---|
  | global RPA | 3 ms | 610 ms (79%) |
  | expert gmm | 51 ms | 50 ms |
  | MoE a2a (dispatch / combine / wait) and other XLA ops | ~80 ms | ~82 ms |
  | all-gather | 10.7 ms | 10.5 ms |
  | SWA RPA | 6.9 ms | 8.7 ms |

  Long context is attention. The 11-10 PP8 analysis put the RPA at 195.5 bundles per chunk against a 192-bundle MXU-pop / EUP floor. The short-context headroom is the MoE.
- **The fp4 expert gmm is VALU-bound by per-block partial rescaling.** gmm_v2 runs fp4 x fp8 with block-256 scales on both sides, so K = 4096 makes 16 partials per [tm x tn]. Each partial is popped, cast, multiplied by the lhs scale and the rhs scale, and accumulated: about 20k of 24k VALU ops per 256x4096x2048 step. The step is 7298 bundles, about 3.7 us; about 36 steps make one w13 call (134 us), matching xprof, so the kernel does not stall. fp4 unpack is 17% of the VALU, and the 4 MiB weight DMA per step is about 30% of compute. Layers 0 and 61 run an fp8 variant at 4601 bundles per step.
- **Why the gmm tiles did not pay** (ctx 0, baseline 157.7 ms):
  - **tm=128: 163.6 ms** (trace + LLO `20261116-045918-99de`). Per step, the MXU work and the partial pops halve (vmatmul 1012 -> 490, vpop 5521 -> 2944), but the fp4 unpack does not (1825 -> 1667): every step unpacks a whole 4096 x 2048 weight tile whatever tm is. With about 128 rows per expert plus the 16-row alignment offset, most experts take two steps, and the second step unpacks the same weights again (they are not re-DMA'd). The result is 1.72x the steps at 0.60x the bundles: +4% per call.
  - **tm=512 / tn=1024: 200.7 ms.** Rows per expert are about 128, so tm=512 pads 3/4 of the MXU work, and tn=1024 doubles the n-steps, each of which re-reads and re-quantizes the lhs.
  - **2 or 8 MoE chunks instead of 4: 164.4 / 166.2 ms.** Two chunks hide less of the a2a behind the expert FFN. Eight chunks make smaller gmm calls with more fixed cost per call, and more unpack-block overlap. These two causes are inferred, not yet traced.
- **Fix in progress: requantize to e4m3 in VMEM with per-channel scales, and one lhs scale per row.** The MXU then accumulates the whole K in one partial: about 1.5k rescale ops per step instead of about 20k, plus about 12k requant ops. The LLO-based estimate is w13 134 -> 80-85 us and w2 about halved, a chunk about 158 -> 133-139 ms. It needs `gmm_v2`'s `lhs_quant_block_size` (a local, uncommitted, default-unchanged kwarg) and a numerics check: each weight takes one more e4m3 rounding, and blocks far below their column's max scale fall into subnormals. Knob `SGLANG_TPU_MUSE_MK_PF_SP_REQUANT`. The first compile OOMed the anvil compiler; retry `20261116-050753-9026`.
- **`fp8_gather` and W4A16 do not apply to SP + EP.** `SGLANG_TPU_MUSE_MK_PREFILL_FP8_GATHER` sends the TP prefill's xe / u all-gathers in e4m3, and `prefill/spep/program.py` raises if it is set. `w4a16_layers` routes TP prefill's fp4 layers to the fused-RS W4A16 kernel; SP + EP always runs `gmm_v2`. The TP8 reference in the prefill chart comes from 2026-11-10 serving with both on, not from anvil.
