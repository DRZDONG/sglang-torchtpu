# Open bug: UserFatalError at process teardown

Status: OPEN. One root cause found and fixed (MoE weight loading `resize_(0)`);
at least one more source remains.

## Summary

- After a Muse megakernel run finishes (results printed, exit code 0), one or
  more ranks log a TPU `FATAL ERROR ... [USER]` / `UserFatalError` during
  process teardown, after the TPU client starts shutting down.
- Intermittent: roughly 40-60% of runs.
- Seen only at teardown so far. No run has produced wrong results or a mid-run
  fault from this bug. (One earlier mid-run halt was a separate bug, since
  fixed: gmm_v2 on bf16 groups whose start is not on a 16-row tile; see below.)
- Pre-existing: it predates this branch and shows up in every Muse megakernel
  program on the debug box. The SP prefill work did not cause it.

## Symptom (exact log lines)

The fault appears after the TPU client is destroyed:

```
allocator_stats_reporter.cc:147] Stopping AllocatorStatsReporter
E1115 ... async_driver.cc:827] [/dev/vfio/N tpu19:pe0:K] vf_id:V !!!! FATAL ERROR !!!! observed errors are: [USER]. Now taking a TPU core dump...
... async_driver.cc:850] Started collecting core dump...
... no_op_core_dumper.h:70] NoOpCoreDumper::Collect
... Core dump collected.
... No Coroner reporter registered, not sending core dump.
E1115 ... tpu_vxc_driver.cc:3270] TPU encountered a UserFatalError
```

- The "Anomalies" table printed between these lines is empty.
- How faults are counted: `grep -a -c UserFatal <log>` gives 2, 4 or 6 lines
  per faulting run.
- Driver logs on the debug box:
  `/tmp/tpu_logs/<pod>/tpu_driver.<pod>.root.log.{INFO,ERROR}.<date>.<pid>`.
  The ERROR log holds only the lines above, plus RPA scoped-VMEM warnings that
  have nothing to do with this bug:
  `INVALID_ARGUMENT: 67108864 bytes of scoped Vmem requested ... lowering to 67043328`.

## Where it happens

All runs below are on the debug box: single host v7x 2x2x1, 8 ranks, torchrun,
image `sglang-torchtpu:anvil-jax0112-97afaea9`, libtpu 0.0.50,
`SGLANG_TPU_MUSE_MK_QBLOCK=256`.

| Program / run | Faulting runs |
|---|---|
| Pre-refactor tree (PR #2178 tip f35f27c73e, worktree `.claude_scratch/base`), decode megakernel numerics `test_muse_megakernel_numerics.py --layers 2 --global-layers 1 --qblock-ref --served` | 4/4 (2-6 lines each), numerics PASS |
| Current branch, decode megakernel bench `bench_muse_ep_arms.py --arms tp --rows 64` | 3/6 |
| Current branch, prefill TP bench `test_muse_mk_prefill_numerics.py --bench-iters 10 --capture 2 --cases pf_c1q4096_ctx0` | 3/3 before the resize fix (twice); 3/5 after it; 2/6 after it with donation off |
| Prefill TP, small case `--cases pf_c1q384` | 2/4 |
| Prefill SP (`--sp`), 4096 rows | ~1/2 across ~25 runs of variants (see Ruled out) |
| Decode-shaped SP runs (`--sp --cases dec_rN`) | faulted on some R (e.g. 2 of 4 runs) |
| `--setup-only` TP (build weights/pools, exit before any step), after the resize fix | 0/3 |
| `--setup-only` SP | 0/3 |
| `--setup-only` TP, before the resize fix | 1/3 (6 lines) |
| Minimal torchrun script (tpu_dist init, one all_reduce, exit), `.claude_scratch/fault_min.py` | 0/2 |
| Batched RPA unit tests, single process (`pytest test/registered/tpu/unit/test_pallas_rpa_seq_along_lane.py`) | 0/4 |

Conclusion: the bug predates this branch and is common to every Muse
megakernel program on this box.

## Root causes found and fixed

### 1. `untyped_storage().resize_(0)` in MoE weight loading (fixed, commit 856c5dce6d)

- What it was: MoE weight loading freed replaced buffers with
  `old.untyped_storage().resize_(0)`, through the helper
  `_free_replaced_storage` in `sglang_tpu/fp8_moe.py` and
  `sglang_tpu/w4a16_moe.py`. Callers: `fp4_moe.py`, `fp8_moe.py`,
  `w4a16_moe.py`, `expert_loading.py`, `models/muse_spark.py`,
  `muse_megakernel/common/expert_requant.py`.
- Evidence: standalone script `.claude_scratch/fault_served.py`, 8 ranks. It
  runs only the served NVFP4 expert prep `served_expert_refs`
  (`prepare_tpu_nvfp4_moe_weights` + `requant_tpu_nvfp4_experts`), then exits.

  | Variant | Faulting runs |
  |---|---|
  | With `resize_(0)` | 2/3 |
  | `resize_(0)` disabled | 0/4 |
  | Plain expert slabs to device (no prep) | 0/3 |
  | Prepare only (requant skipped) | 1/3 |
  | Full `torch.accelerator.synchronize()` before each `resize_(0)` | 3/3 |

  The sync variant still faults 3/3. So the cause is not in-flight programs
  reading the old buffer: `resize_(0)` by itself breaks teardown.
- Fix: commit 856c5dce6d on branch `junyanxu/muse-mk-layout` (PR #2183). The
  helper is removed. Buffers are now released by dropping the last reference;
  load-before-allocate sites keep no local alias.
- Load-time HBM per rank is the same with and without `resize_(0)`: 11068 MiB
  resident, 12780 MiB peak (`torch.accelerator.memory_allocated` /
  `max_memory_allocated`).
- A related `resize_(0)` hazard was already documented in
  `test/manual/tpu/test_unquant_moe_weight_prep.py`: `copy_or_rebind_param`
  copies in place when the shape is unchanged, so freeing the "old" buffer
  zeroed live weights.
- After this fix, setup-only runs are clean (TP 0/3, SP 0/3), but any run that
  executes a step still faults. A second source remains.

### Other bugs fixed during the hunt (not the teardown cause)

- **Batched RPA zero-length out-of-bounds DMAs.**
  - Bug: in the SEQ_ALONG_LANE layout, `copy_in`/`copy_out` started a DMA for
    every schedule entry. The other layout's q `copy_in` and o/lse `copy_out`
    did the same. Unused entries have size 0 but a stale page index / token
    offset.
  - `--xla_mosaic_on_device_checks=bounds` reported this on the new-KV fetch of
    a 4096-row sliding-window prefill:
    `Operation tpu.memref_slice references out-of-bounds elements in dimension 3 (source location: ... KVBufferedRefSeqAlongLane.copy_in ... bref_override.py:116) HLO: RPAm-p256-b1-q768-k768-sw2048`
  - Fix: commit 1d4c5dab16 guards the three KV DMAs. An uncommitted extension
    guards every DMA start in
    `sglang_tpu/attention/pallas/batched_rpa/bref_override.py` with
    `_start_if(sz, src, dst, sem)`, which starts the DMA only when `sz > 0`.
    Semaphore waits are unchanged; they count valid bytes only.
  - With the guard in place, TP bounds checks are clean but the teardown fault
    remains.
- **gmm_v2 on unaligned bf16 groups.** gmm_v2 faults (mid-run UserFatalError or
  hang) on bf16 groups whose start is not 16-row aligned. Rather than fixing
  gmm_v2, the SP MoE pads expert groups to 16 rows (`GROUP_ALIGN` in
  `common/a2a.py`).

## Ruled out

Each variant still faulted. 4 runs each unless noted.

| Hypothesis | Variant | Result |
|---|---|---|
| RPA separate_out (e4m3 q, separate bf16 output buffer, SP only) | q kept bf16 so separate_out is off (`SP_DBG_QBF16`) | 2/4 fault |
| Streamed a2a returning DMA semaphores as kernel outputs across kernels | One synchronous send+wait kernel with scratch semaphores (`SP_DBG_SYNC_A2A`) | 1/4 |
| Barrier-semaphore collective ids 11/12 of the SP a2a kernels | ids 9/10 (`SP_DBG_CID=9,10`) | 2/4 |
| XLA all_to_all of the o-proj row exchange | o-proj RS by row instead (`SGLANG_TPU_MUSE_MK_SP_O_RS=1`) | 1/4 |
| unpack_rows Pallas kernel (clamped block revisits) | Plain-XLA decode (`SP_DBG_XLA_UNPACK`) | 1/4 |
| KV pool donation / `input_output_aliases` in `muse_megakernel/ops.py` `_export_program` | Donation and aliases disabled (`SP_DBG_NODONATE`), TP | 2/6 |
| RPA zero-length OOB DMAs | Guarded (see above) | TP 3/5; SP still faults |
| Batched RPA alone | RPA unit tests, single process | 0/4 (not decisive: small configs, 1 process) |

- The `SP_DBG_*` switches were temporary and have been removed from the code.
  `SGLANG_TPU_MUSE_MK_SP_O_RS` is a real switch.
- Mosaic semaphore checks (`--xla_tpu_mosaic_enable_semaphore_checks=true`):
  2 SP runs, no semaphore report. Neither run faulted, so this is not
  conclusive.

## Tools and how to use them

| Tool | Usage | Gotchas |
|---|---|---|
| Mosaic on-device checks | `LIBTPU_INIT_ARGS="--xla_mosaic_on_device_checks=bounds --xla_tpu_mosaic_enable_semaphore_checks=true"`. A violation halts with `Detailed error: Node N halted unexpectedly ... Operation ... references out-of-bounds ...` plus the source location. | Bounds checks use extra VMEM. TP fits. SP fails to compile: the gmm_v2 tile (256,4096,2048) needs 72 MiB of 64. Even with tiles (128,1024,1024), the RPA mixed kernel `RPAm-p256-b1-q1024-k1024-sw2048` is 352 KiB over. |
| In-kernel trace regions | `SGLANG_TPU_MK_TRACE_REGIONS=1` plus libtpu `--xla_enable_custom_call_region_trace=true` and device tracer level > 1. Regions land on the "XLA TraceMe" line. | `sglang_tpu/profiler.py` pins `device_tracer_level=1`. `test/manual/tpu/bench_muse_ep_arms.py` patches it to 2 when the env is set. |
| xprof CLI | `/tmp/xprof_venv/bin/xprof create_events_db --session_id=<run dir>` writes parquet. `check_kernel_profiling` lists planes/lines. | |
| py-spy | Installed in the container. | `py-spy dump` fails with `Permission denied (os error 13)` (no ptrace). |
| Core dumps | | NoOp (no Coroner reporter registered), so no device state is captured. |

Operational gotchas:

- Back-to-back torchrun runs can fail with `START_SESSION failed` because the
  previous processes still hold the TPU. Sleep 20-30 s between runs. Count those
  runs as invalid, not as fault or no-fault.
- Killing processes in the container: match exact cmdlines and exclude your
  own shell PID (`$$`). Otherwise the kill loop kills itself.
- Debug pods get rescheduled (FailedScheduling -> Scheduled) or hit their
  active deadline, and /tmp is wiped when that happens. Read results promptly.

## How to reproduce

On a debug box (`cdk_ssh <job> server -- bash -c ...`), from `/workspace/sglang`:

```bash
eval $(python3 -m torch.tpu.distributed)
export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE SGLANG_TPU_MUSE_MK_QBLOCK=256
unset TPU_VMODULE; export TPU_STDERR_LOG_LEVEL=2
for i in 1 2 3 4; do sleep 30
  torchrun --nproc_per_node="$WORLD_SIZE" test/manual/tpu/bench_muse_ep_arms.py --arms tp --rows 64 > /tmp/f_$i.log 2>&1
  echo "run $i exit $? fatal $(grep -a -c UserFatal /tmp/f_$i.log)"
done
```

To run another program, swap the torchrun script and arguments:

| Program | Script and args |
|---|---|
| Prefill TP | `test/manual/tpu/test_muse_mk_prefill_numerics.py --bench-iters 10 --capture 2 --cases pf_c1q4096_ctx0` |
| Prefill SP | same, add `--sp` |
| Setup only | same, add `--setup-only` |
| Small TP case | `test/manual/tpu/test_muse_mk_prefill_numerics.py --cases pf_c1q384` |
| Pre-refactor decode | `test_muse_megakernel_numerics.py --layers 2 --global-layers 1 --qblock-ref --served` (in the `.claude_scratch/base` worktree) |

A fault shows up as a non-zero `fatal` count, even though `exit` is 0.

## Open hypotheses and next steps

Hypotheses (none confirmed):

- H1: a second source sits on the execution path that every program shares.
  Setup-only runs are clean after the resize fix; any run that executes a step
  faults. Candidates not yet isolated:
  - the torch_tpu jax-export path: `_CpuExportJaxCallable` from
    `muse_megakernel/ops.py` `_export_program`, i.e. a shard_map program with
    Pallas custom calls;
  - XLA collectives inside the exported program;
  - the teardown order of the torch_tpu client vs. the distributed process
    group.

Next experiments:

1. Build a minimal exported shard_map program with one trivial Pallas kernel,
   and separately one with an XLA all_gather. Run each through the same torch
   op path, N=8 runs.
2. Call `dist.destroy_process_group()` and/or an explicit sync before exit, to
   see whether teardown order matters.
3. Set `TPU_STDERR_LOG_LEVEL=0` and read the driver INFO log around the fault
   for the failing node id.
4. Check whether the faulting vfio device / rank is the same across runs.
5. Try a newer libtpu / torch_tpu image.
6. Escalate to the torch_tpu / libtpu owners with the minimal repro.

## Environment

- Branch `junyanxu/muse-mk-layout`, PR #2183 (draft, stacked on #2178).
- Commits: 1d4c5dab16 (RPA guard), 856c5dce6d (`resize_(0)` removal),
  9f900b087b (SP prefill). Later work is uncommitted.
- Hardware/software: single host v7x 2x2x1, 8 ranks, torchrun, image
  `sglang-torchtpu:anvil-jax0112-97afaea9`, libtpu 0.0.50,
  `SGLANG_TPU_MUSE_MK_QBLOCK=256`.

## Artifacts

| Kind | Location |
|---|---|
| Repro scripts | `.claude_scratch/fault_*.sh`, `.claude_scratch/fault_served.py`, `.claude_scratch/fault_min.py`, `.claude_scratch/a2a_roundtrip.py` (under the repo root `/home/junyanxu_google_com/nfs/code/sglang-muse-disagg-kernel/`) |
| Pre-refactor worktree | `.claude_scratch/base` (PR #2178 tip f35f27c73e) |
| Bench scripts | `test/manual/tpu/bench_muse_ep_arms.py`, `test/manual/tpu/test_muse_mk_prefill_numerics.py` |
| Run logs | `/tmp/f_<i>.log` on the debug box (wiped when the pod is rescheduled) |
| Driver logs | `/tmp/tpu_logs/<pod>/tpu_driver.<pod>.root.log.{INFO,ERROR}.<date>.<pid>` on the debug box |
| Debug boxes | cdk jobs from recipe `agentx-muse-spark-singlehost-sglang-junyanxu`, `DEBUG_IDLE=true`: `j-3940d541-a806-481c-bcce`, `j-1841ccff-96b4-42c3-bb33`. `j-01309861-0c05-41ed-b590` has expired. |
