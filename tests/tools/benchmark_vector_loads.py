"""Synthetic grouped-load benchmark (not a submission-wide speedup claim).

    .venv/bin/python tests/tools/benchmark_vector_loads.py          # offline checks
    .venv/bin/python tests/tools/benchmark_vector_loads.py --remote # RTX runner sweep

Each thread sums WIDTH consecutive floats and writes one result. Scalar and
vector variants differ only in load width. Warmed repeated buffers may fit L2.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from benchmark import check_outputs
from compile import Compiler
from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import Var
from submit import RUNNER, post


class ScalarCompiler(Compiler):
    def load_alignment(self, name, index):
        return 4  # Force exactly the original scalar load instructions.


def make_case(rows, width):
    source = ROOT / 'tests/problems/p00_scalar_add'
    data = json.loads(next(source.glob('*.ir.json')).read_text())
    for name, node in data['nodes'].items():
        node['outputs'][0]['shape'] = [rows if name == 'add' else rows * width]
    graph = Graph.from_dict(data)
    node = graph.nodes['add']
    tile = node.op.body[0]
    load, _, add, *rest = tile.body
    axis = replace(tile.axes[0], extent=type(tile.axes[0].extent)(rows))
    index = Var(axis.name)
    names = tuple(f'x{i}' for i in range(width))
    body = [replace(load, names=names, index=(index * width,))]
    total = names[0]
    for i in range(1, width):
        result = f'sum{i}'
        body.append(replace(add, name=result, args=(total, names[i])))
        total = result
    body.append(replace(rest[-1], index=(index,), values=(total,)))
    kernel = replace(node.op, name='grouped_sum', body=(replace(tile, axes=(axis,), body=tuple(body)),))
    launch = dict(arg_order=['x0', 'add'], grid=[(rows + 255) // 256, 1, 1],
                  block=[256, 1, 1], smem_bytes=0, zero_outputs=[],
                  args=[dict(name='x0', dtype='f32', shape=[rows * width], seed=42),
                        dict(name='add', dtype='f32', shape=[rows], output=True)])
    x = np.random.default_rng(42).standard_normal(rows * width).astype(np.float32).reshape(rows, width)
    expected = x[:, 0].copy()
    for i in range(1, width):
        expected += x[:, i]
    scalar = ScalarCompiler(graph, kernel, launch).compile()
    vector = Compiler(graph, kernel, launch).compile()
    assert scalar.count('ld.global.f32') == width
    assert vector.count('ld.global.v4.f32') == width // 4
    # All non-load instructions, including stores and arithmetic, must match.
    strip_loads = lambda ptx: [line for line in ptx.splitlines() if 'ld.global.' not in line]
    assert strip_loads(scalar) == strip_loads(vector)
    return launch, {'add': expected}, dict(scalar=scalar, vector=vector)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--rows', nargs='+', type=int, default=[4096, 65536, 262144])
    parser.add_argument('--widths', nargs='+', type=int, choices=[4, 8, 16], default=[4, 16])
    parser.add_argument('--trials', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path('/tmp/emmy-vector-synthetic.json'))
    args = parser.parse_args()
    if not 2 <= args.trials <= 20 or any(not 1 <= n <= 1048576 for n in args.rows):
        parser.error('require trials=2..20 and rows=1..1048576')
    source = (ROOT / 'compile.py').read_bytes()
    report = dict(started_at=datetime.now(timezone.utc).isoformat(),
                  compiler_sha256=hashlib.sha256(source).hexdigest(),
                  synthetic=True, warmup=20, iters=200, trials=args.trials, cases=[])
    if args.remote:
        health = requests.get(RUNNER + '/health', timeout=20)
        health.raise_for_status()
        report['health'] = health.json()
        assert report['health'].get('can_compile') and report['health'].get('can_run')
    for width in args.widths:
        for rows in args.rows:
            launch, expected, variants = make_case(rows, width)
            if not args.remote:
                print(f'PASS offline: rows={rows}, width={width}', flush=True)
                continue
            case = dict(rows=rows, width=width, input_bytes=rows * width * 4,
                        samples_us={'scalar': [], 'vector': []}, registers={}, max_abs_error=0)
            binaries = {}
            for label, ptx in variants.items():
                compiled = post(RUNNER, '/compile', {'ptx': ptx})
                assert compiled.get('ok'), compiled
                binaries[label] = compiled['cubin_id']
                case['registers'][label] = compiled.get('registers')
            for trial in range(args.trials):
                order = ('scalar', 'vector') if trial % 2 == 0 else ('vector', 'scalar')
                for label in order:
                    for _ in range(30):
                        run = post(RUNNER, '/run', dict(
                            cubin_id=binaries[label], entry='grouped_sum',
                            **{k: launch[k] for k in ('grid', 'block', 'smem_bytes', 'zero_outputs', 'args')},
                            warmup=report['warmup'], iters=report['iters'], max_return=max(4096, rows)))
                        if run.get('stage') != 'busy':
                            break
                        time.sleep(2)
                    assert run.get('ok') and run['iters'] == report['iters'], run
                    error = check_outputs(expected, run['outputs'], rtol=0, atol=0)
                    case['max_abs_error'] = max(error, case['max_abs_error'])
                    latency = float(run['median_us'])
                    assert np.isfinite(latency) and latency > 0
                    case['samples_us'][label].append(latency)
                print(f'rows={rows} width={width} trial={trial + 1}: '
                      f'scalar={case["samples_us"]["scalar"][-1]} '
                      f'vector={case["samples_us"]["vector"][-1]} us', flush=True)
            case['median_us'] = {k: statistics.median(v) for k, v in case['samples_us'].items()}
            case['speedup'] = case['median_us']['scalar'] / case['median_us']['vector']
            case['vector_wins'] = sum(v < s for s, v in zip(case['samples_us']['scalar'], case['samples_us']['vector']))
            report['cases'].append(case)
            assert source == (ROOT / 'compile.py').read_bytes(), 'Compiler changed during benchmark'
            report['finished_at'] = datetime.now(timezone.utc).isoformat()
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(f'RESULT rows={rows} width={width}: {case["speedup"]:.3f}x, '
                  f'{case["vector_wins"]}/{args.trials} vector wins', flush=True)
    if args.remote:
        print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
