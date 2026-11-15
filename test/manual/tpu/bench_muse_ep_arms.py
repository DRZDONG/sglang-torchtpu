# SPDX-License-Identifier: Apache-2.0
"""Decode EP paradigm sweep: gather (EP, every row on every rank) vs a2a (SP + EP, own rows
behind ragged all-to-alls), same weights and step inputs, at each row count.

Builds weights with test_muse_megakernel_numerics.py's builders (--ep layout). Run one process
per device:

    eval $(python3 -m torch.tpu.distributed)
    export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE
    SGLANG_TPU_MUSE_MK_QBLOCK=256 torchrun --nproc_per_node="$WORLD_SIZE" \\
        test/manual/tpu/bench_muse_ep_arms.py --rows 16 32 64 128 256 --profile-dir /tmp/ep_xprof

Rank 0 prints one MK_EP_BENCH json line per (rows, arm). --profile-dir captures an xprof trace
of --profile-iters steps per (rows, arm); add SGLANG_TPU_MK_TRACE_REGIONS=1 for per-phase regions.
Looping one input set is fine: every step streams hundreds of MiB to GiB of weights from HBM,
far beyond on-chip memory, so there is no warm-cache bias to flush.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_muse_megakernel_numerics as N  # noqa: E402


def _compiled_step(runner, case, st, kv):
    """One traced + compiled step closure (the arm is fixed at trace time)."""
    d = runner.device
    experts = runner.experts(case.expert_group)
    meta = torch.tensor([st.n_live, runner.rank], dtype=torch.int32, device=d)
    capture = list(runner.capture)
    op = torch.ops.sglang_tpu_pallas.muse_megakernel

    def run(meta, pos, pf, psw, emb, weights, experts, kv):
        return op(
            meta, pos, pf, psw, emb, weights, experts, kv, case.q, case.n_req, capture
        )

    torch._dynamo.reset()
    # the executable cache keys on the graph, not on the arm the op picks while tracing
    torch.tpu._clear_cache()
    compiled = torch.compile(run, fullgraph=True, dynamic=False)
    args = (meta, st.pos.to(d), st.pf.to(d), st.psw.to(d), st.emb.to(d))
    return lambda: compiled(*args, runner.weights, experts, kv)


def bench(runner, rows: int, arm: str, args) -> dict:
    os.environ["SGLANG_TPU_MUSE_MK_EP_ARM"] = arm
    case = N.Case(f"bench_r{rows}", args.kv, rows, 1, (args.ctx,) * rows)
    st = N.step_inputs(case)
    kv = runner.pools(case, st)
    runner.overrides, runner.last_cfg = {}, None
    step = _compiled_step(runner, case, st, kv)
    hid, _ = step()
    hid.cpu()
    cfg = runner.last_cfg
    if cfg is None:
        raise RuntimeError("the step ran a cached executable: the op was not traced")
    for _ in range(args.warmup):
        hid, _ = step()
    hid.cpu()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        hid, _ = step()
    hid.cpu()
    step_ms = (time.perf_counter() - t0) * 1e3 / args.iters
    if args.profile_dir:
        from pathlib import Path

        import sglang_tpu.profiler as prof
        from sglang_tpu.profiler import start_tpu_profile, stop_tpu_profile

        if os.environ.get("SGLANG_TPU_MK_TRACE_REGIONS", "0") != "0":
            # in-kernel regions need a device tracer level above 1 (profiler.py pins 1)
            orig = prof._tpu_profile_options

            def _opts(**kw):
                o = orig(**kw)
                if o is not None:
                    o.device_tracer_level = 2
                return o

            prof._tpu_profile_options = _opts

        out = Path(args.profile_dir) / f"{arm}_r{rows}" / f"rank{runner.rank}"
        dist.barrier()
        trace = start_tpu_profile(output_dir=out, with_cpu=False)
        for _ in range(args.profile_iters):
            hid, _ = step()
        hid.cpu()
        stop_tpu_profile(trace)
    return {
        "rows": rows,
        "padded_rows": cfg.rows,
        "arm": arm,
        "layers": runner.n_layers,
        "ctx": args.ctx,
        "step_ms": round(step_ms, 4),
        "ms_per_layer": round(step_ms / runner.n_layers, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rows", type=int, nargs="+", default=[16, 32, 48, 64, 96, 128, 192, 256]
    )
    # "tp" alone: the decode TP megakernel baseline (TP weight layout, served QBLOCK experts)
    ap.add_argument("--arms", nargs="+", default=["gather", "a2a"])
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--global-layers", default="1,5")
    ap.add_argument("--ctx", type=int, default=4096, help="cached tokens per request")
    ap.add_argument("--kv", default="sal_fp8")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--profile-dir", default="")
    ap.add_argument("--profile-iters", type=int, default=3)
    ap.add_argument("--out", default="/tmp/mk_ep_bench.json")
    # aux (capture) layers, comma-separated; empty (default) = none. Serving captures a few of
    # 62 layers; capturing every layer makes EP all-gather every row each layer (TP gets it free)
    ap.add_argument("--capture", default="")
    args = ap.parse_args()
    tp_only = args.arms == ["tp"]
    if "tp" in args.arms and not tp_only:
        raise ValueError(
            "--arms tp runs alone (TP and EP need different weight layouts)"
        )
    N._EP = not tp_only
    # the served path: QBLOCK-requantized fp4 experts, as production loads them
    N._QBLOCK_REF = N._SERVED = tp_only

    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    if dist.get_world_size() != N.TP:
        raise RuntimeError(f"needs {N.TP} processes, got {dist.get_world_size()}")
    torch.set_num_threads(max(1, (os.cpu_count() or 8) // N.TP))
    torch._dynamo.config.cache_size_limit = 256
    torch.compiler.config.force_disable_caches = True
    from sglang_tpu.muse_megakernel import ops

    global_layers = tuple(int(x) for x in args.global_layers.split(",") if x)
    max_position = -(-(args.ctx + 16) // 8) * 8
    runner = N._Runner(ops, args.layers, global_layers, rank, max_position)
    runner.capture = [int(x) for x in args.capture.split(",") if x]
    dist.barrier()
    results = []
    for rows in args.rows:
        for arm in args.arms:
            r = bench(runner, rows, arm, args)
            results.append(r)
            if rank == 0:
                print("MK_EP_BENCH " + json.dumps(r), flush=True)
            dist.barrier()
    if rank == 0:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    print(f"MK_EP_BENCH_DONE rank={rank}", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
