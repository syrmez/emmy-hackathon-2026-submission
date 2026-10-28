"""Refresh performance.html with measured, correctness-checked results.

    .venv/bin/python tests/benchmark.py --include-local

No web server or frontend dependencies. The page embeds its own data snapshot.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import requests

from emmy.compiler.graph import Graph
from submit import PROBLEMS, RUNNER, not_ptx, post

ROOT = Path(__file__).resolve().parents[1]
DATA_TAG = '<script id="results" type="application/json">'
CATEGORIES = dict(p="Pointwise", r="Reduction", m="Matmul", f="Fused", a="Attention")


def check_outputs(expected, outputs, rtol, atol):
    """A timing is only accepted if every reference output was returned correctly."""
    if not expected:
        raise ValueError("no reference outputs; correctness cannot be checked")
    worst = 0.0
    for name, ref in expected.items():
        out = outputs.get(name)
        if out is None or out.get("truncated"):
            raise ValueError(f"missing or truncated output: {name}")
        got = np.asarray(out["values"], dtype=ref.dtype)
        if got.size != ref.size or not np.isfinite(got).all():
            raise ValueError(f"wrong size or non-finite output: {name}")
        got = got.reshape(ref.shape)
        error = float(np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64)))) if got.size else 0.0
        if not np.allclose(got, ref, rtol=rtol, atol=atol):
            raise ValueError(f"{name}: max abs error {error:.6g}; rtol={rtol}, atol={atol}")
        worst = max(worst, error)
    return worst


def measure(folder, dataset, args):
    row = dict(problem=folder.name, dataset=dataset, category=CATEGORIES.get(folder.name[0], "Generated"),
               status="error", gpu_us=None, compile_ms=None, registers=None, error=None)
    try:
        meta = json.loads((folder / "meta.json").read_text())
        # ponytail: all current cases have one kernel; add chaining when a case needs it.
        if len(meta["kernels"]) != 1:
            raise ValueError("this benchmark currently supports single-kernel problems only")
        kernel = meta["kernels"][0]
        if any(a.get("intermediate") for a in kernel["args"]):
            raise ValueError("intermediate inputs require kernel chaining")
        row.update(expression=meta.get("code", ""), grid=kernel["grid"], block=kernel["block"],
                   smem_bytes=kernel["smem_bytes"], entry=kernel["entry"])
        with np.load(folder / "expected.npz", allow_pickle=False) as archive:
            expected = dict(archive)
        if set(expected) != set(meta["graph_outputs"]):
            raise ValueError("references do not cover all graph outputs")
        ir = folder / f'{kernel["entry"]}.ir.json'
        compile_ms = []
        for _ in range(args.trials):
            start = time.perf_counter()
            result = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir)],
                                    cwd=ROOT, capture_output=True, text=True, timeout=20)
            compile_ms.append((time.perf_counter() - start) * 1000)
            if result.returncode:
                raise ValueError(result.stderr.strip() or "compiler failed without diagnostics")
            ptx = result.stdout
            if why := not_ptx(ptx):
                raise ValueError(f"compiler output is not PTX: {why}")
        row.update(compile_ms=statistics.median(compile_ms), compile_samples_ms=compile_ms,
                   ptx_bytes=len(ptx.encode()), ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest())
        compiled = post(args.server, "/compile", {"ptx": ptx})
        if not compiled.get("ok") or kernel["entry"] not in compiled.get("entry_points", []):
            raise ValueError(f'assembly/entry failure: {compiled.get("stderr", compiled)}')
        row["registers"] = compiled.get("registers", {}).get(kernel["entry"])
        need = sum(math.prod(a["shape"]) for a in kernel["args"] if a.get("output"))
        if need > 2**22:
            raise ValueError("outputs exceed the runner's full-return limit")
        gpu_us, errors = [], []
        for _ in range(args.trials):
            for _ in range(30):
                run = post(args.server, "/run", {
                    "cubin_id": compiled["cubin_id"], "entry": kernel["entry"],
                    "grid": kernel["grid"], "block": kernel["block"], "smem_bytes": kernel["smem_bytes"],
                    "zero_outputs": kernel["zero_outputs"], "args": kernel["args"],
                    "warmup": args.warmup, "iters": args.iters, "max_return": max(4096, need),
                })
                if run.get("stage") != "busy":
                    break
                time.sleep(2)
            if not run.get("ok"):
                raise ValueError(f'runner failure: {run.get("stderr", run)}')
            errors.append(check_outputs(expected, run["outputs"], meta["rtol"], meta["atol"]))
            latency = float(run["median_us"])
            if not math.isfinite(latency) or latency <= 0 or run["iters"] != args.iters:
                raise ValueError("runner returned invalid timing metadata")
            gpu_us.append(latency)
        row.update(status="pass", gpu_us=statistics.median(gpu_us), gpu_samples_us=gpu_us,
                   max_abs_error=max(errors))
        if folder.name == "p00_scalar_add" and dataset == "public":
            graph = Graph.from_dict(json.loads(ir.read_text()))
            tile = graph.nodes[kernel["node"]].op.body[0]
            row["walkthrough"] = dict(
                cells=tile.n_elements, threads=math.prod(kernel["block"]),
                elements=math.prod(expected[meta["graph_outputs"][0]].shape),
                arg_order=kernel["arg_order"], ir=ir.with_suffix(".txt").read_text(), ptx=ptx,
            )
    except (Exception, SystemExit) as exc:
        row.update(status="error", error=str(exc) or type(exc).__name__)
    return row


def render_snapshot(template, report):
    if template.count(DATA_TAG) != 1:
        raise ValueError("expected one embedded results block in performance.html")
    start = template.index(DATA_TAG) + len(DATA_TAG)
    end = template.index("</script>", start)
    # Prevent a source expression or diagnostic from closing the script element.
    data = json.dumps(report, allow_nan=False).replace("<", "\\u003c")
    return template[:start] + "\n" + data + "\n" + template[end:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-local", action="store_true", help="also measure generated local_problems")
    parser.add_argument("--trials", type=int, default=3, help="independent batches and CLI invocations (1..10)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--server", default=RUNNER)
    parser.add_argument("--output", type=Path, default=ROOT / "tests/performance.html")
    args = parser.parse_args()
    if not (1 <= args.trials <= 10 and 0 <= args.warmup <= 20 and 1 <= args.iters <= 200):
        parser.error("require trials=1..10, warmup=0..20, iters=1..200")
    args.server = args.server.rstrip("/")
    response = requests.get(f"{args.server}/health", timeout=20)
    response.raise_for_status()
    health = response.json()
    if not health.get("can_compile") or not health.get("can_run"):
        sys.exit("runner cannot compile and execute kernels right now")
    compiler = ROOT / "compile.py"
    fingerprint = hashlib.sha256(compiler.read_bytes()).hexdigest()
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), compiler_sha256=fingerprint,
                  python=platform.python_version(), host=platform.platform(), gpu=health.get("gpu", "unknown"),
                  arch=health.get("default_arch", "unknown"), server=args.server,
                  trials=args.trials, warmup=args.warmup, iters=args.iters, rows=[], walkthrough=None)
    folders = [(p.parent, "public") for p in sorted(PROBLEMS.glob("*/meta.json"))]
    if args.include_local:
        local = sorted((ROOT / "tests/local_problems").glob("*/meta.json"))
        if not local:
            parser.error("--include-local requires tests/local_problems; see tests/README.md")
        folders += [(p.parent, "generated") for p in local]
    for folder, dataset in folders:
        row = measure(folder, dataset, args)
        if "walkthrough" in row:
            report["walkthrough"] = row.pop("walkthrough")
        report["rows"].append(row)
        timing = f'{row["gpu_us"]:.3f} us' if row["status"] == "pass" else row["error"]
        print(f'{row["status"].upper():5} {folder.name:25} {timing}', flush=True)
    if hashlib.sha256(compiler.read_bytes()).hexdigest() != fingerprint:
        sys.exit("compiler changed during measurement; refusing to save a mixed snapshot")
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    template = (ROOT / "tests/performance.html").read_text()
    args.output.write_text(render_snapshot(template, report))
    print(f"Saved {args.output}")
    return int(any(row["status"] != "pass" for row in report["rows"]))


if __name__ == "__main__":
    sys.exit(main())
