"""Small semantic adversaries beyond the generated problem suite.

    .venv/bin/python test_edge_cases.py
    .venv/bin/python test_edge_cases.py --remote

References use independent NumPy calculations, not another IR compiler.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np

from benchmark import check_outputs
from compile import Compiler
from emmy.compiler.graph import Graph
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import CpAsyncCommit, CpAsyncCopy, CpAsyncWait, Smem, Sync, WarpShuffle
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Init, Let, Load, Loop, StridedLoop, Write
from submit import RUNNER, post

ROOT = Path(__file__).resolve().parent
SOURCE = json.loads(next((ROOT / 'problems/p00_scalar_add').glob('*.ir.json')).read_text())
TILE = Graph.from_dict(SOURCE).nodes['add'].op.body[0]
ADD = ElementwiseImpl('add')
A = Var('a')


def axis(name, size):
    return replace(TILE.axes[0], name=name, extent=type(TILE.axes[0].extent)(size))


def cases():
    x = np.random.default_rng(0).standard_normal(128).astype(np.float32)
    load = Load(name='x', input='x0', index=(A,))
    store = Write(output='add', index=(A,), values=('x',))
    yield 'integer_let', 64, (
        Let('offset', Literal(1, 'int')), Let('pos', A * 2 + Var('offset')),
        replace(load, index=(Var('pos'),)), store), x[1::2], 0
    yield 'index_branch_scope', 64, (
        Let('offset', Literal(0, 'int')),
        Cond(BinaryExpr('<', A, Literal(32, 'int')), (Let('offset', Literal(5, 'int')),)),
        replace(load, index=(A + Var('offset'),)), store), x[:64], 0
    yield 'float_predicate', 128, (
        load, Cond(BinaryExpr('>', Var('x'), Literal(0.0)), (store,),
                   (Let('zero', 0.0), replace(store, values=('zero',))))), np.maximum(x, 0), 0
    yield 'float_let', 128, (
        load, Let('result', Var('x') * Literal(2.0) + Literal(1.0)),
        replace(store, values=('result',))), x * np.float32(2) + np.float32(1), 0
    for op, expected in (('negative', -x), ('square', x * x), ('abs', np.abs(x))):
        yield op, 128, (load, Assign('y', ElementwiseImpl(op), ('x',)),
                       replace(store, values=('y',))), expected, 0
    indices = np.arange(128)
    masked = BinaryExpr('|', BinaryExpr('<<', BinaryExpr('^', BinaryExpr('&', A, Literal(7, 'int')),
                        Literal(3, 'int')), Literal(1, 'int')), Literal(1, 'int'))
    yield 'bitwise_index', 128, (replace(load, index=(masked,)), store), x[((indices & 7) ^ 3) << 1 | 1], 0
    shifted = BinaryExpr('>>', A - 128, Literal(2, 'int')) + 32
    yield 'signed_shift', 128, (replace(load, index=(shifted,)), store), x[((indices - 128) >> 2) + 32], 0
    selected = TernaryExpr(BinaryExpr('>', Var('x'), Literal(0.0)), A, Literal(127, 'int') - A)
    yield 'integer_ternary_float_condition', 128, (
        load, Let('pos', selected), replace(load, names=('y',), index=(Var('pos'),)),
        replace(store, values=('y',))), x[np.where(x > 0, indices, 127 - indices)], 0
    special = TernaryExpr(BinaryExpr('==', A, Literal(0, 'int')), Literal(float('nan')),
              TernaryExpr(BinaryExpr('==', A, Literal(1, 'int')), Literal(0.0),
              TernaryExpr(BinaryExpr('==', A, Literal(2, 'int')), Literal(-1.0), Literal(1.0))))
    for name, predicate in (('float_truth', Var('special')),
                            ('float_not_equal_nan', BinaryExpr('!=', Var('special'), Literal(0.0)))):
        yield name, 4, (
            Let('special', special), Let('one', 1.0), Let('zero', 0.0),
            Cond(predicate, (replace(store, values=('one',)),),
                 (replace(store, values=('zero',)),))), np.array([1, 0, 1, 1], dtype=np.float32), 0
    yield 'explicit_loop_end', 64, (
        Let('k_end', Literal(0, 'int')),
        StridedLoop(axis('k', 8), start=Literal(0, 'int'), step=1, end=Literal(2, 'int'), body=(
            replace(load, index=(A * 2 + Var('k_end') - 1,)), Accum('total', 'x'))),
        replace(load, index=(A + Var('k_end'),)), Assign('result', ADD, ('total', 'x')),
        replace(store, values=('result',))), x[1::2] * np.float32(2) + x[:64], 0
    total = x.reshape(16, 8)[:, 0].copy()
    for i in range(1, 8):
        total += x.reshape(16, 8)[:, i]
    yield 'nested_unseeded_loop', 16, (
        Init('total', 0.0, dtype=TILE.body[0].dtype), Loop(axis('j', 4), (
            Loop(axis('k', 2), (
                replace(load, index=(A * 8 + Var('j') * 2 + Var('k'),)),
                Accum('total', 'x')), seed=False),), seed=False),
        replace(store, values=('total',))), total, 0
    for length in (8, 16, 32):
        combine = (Assign('x', ADD, ('x', 'other')),)
        shuffle = WarpShuffle(('x',), ('other',), combine, length)
        expected = np.repeat(x.reshape(-1, length).sum(axis=1), length)
        yield f'warp_{length}', 128, (load, shuffle, store), expected, 0
    for nbytes in (4, 8, 16):
        width = nbytes // 4
        names = tuple(f'x{i}' for i in range(width))
        body = (Smem('scratch', (128,), align=16),
                CpAsyncCopy('scratch', (A * width,), 'x0', (A * width,), nbytes=nbytes),
                CpAsyncCommit(), CpAsyncWait(0), Sync(),
                Load(names=names, input='scratch', index=(A * width,)),
                Write(output='add', index=(A * width,), values=names))
        for smem in (0, 512):
            yield f'async_{nbytes}_{"dynamic" if smem else "static"}', 128 // width, body, x, smem
    yield 'atomic_repeated_launch', 128, (
        load, replace(store, index=(A % 4,), atomic=True)), x.reshape(-1, 4).sum(axis=0), 0
    # Addresses 1,5,9,... are not 16-byte aligned; do not widen these loads.
    names = ('p', 'q', 'r')
    yield 'unaligned_group_tail', 32, (
        Load(names=names, input='x0', index=(A * 4 + 1,)),
        Write(output='add', index=(A * 3,), values=names)), x.reshape(-1, 4)[:, 1:].ravel(), 0


def main(remote=False):
    failures = []
    for name, threads, body, expected, smem in cases():
        try:
            data = json.loads(json.dumps(SOURCE))
            data['nodes']['add']['outputs'][0]['shape'] = [expected.size]
            graph = Graph.from_dict(data)
            kernel = replace(graph.nodes['add'].op, name=name,
                             body=(replace(TILE, axes=(axis('a', threads),), body=body),))
            # Async cases use one full cooperative block, never an early-exiting lane.
            block = threads if name.startswith('async_') else 128
            launch = dict(arg_order=['x0', 'add'], grid=[1, 1, 1], block=[block, 1, 1],
                          smem_bytes=smem, zero_outputs=['add'] if name.startswith('atomic_') else [],
                          args=[dict(name='x0', dtype='f32', shape=[128], seed=0),
                                dict(name='add', dtype='f32', shape=[expected.size], output=True)])
            ptx = Compiler(graph, kernel, launch).compile()
            if name == 'unaligned_group_tail':
                assert 'ld.global.v' not in ptx
            if name == 'float_predicate':
                assert 'setp.gt.f32' in ptx
            if remote:
                compiled = post(RUNNER, '/compile', {'ptx': ptx})
                assert compiled.get('ok'), compiled
                for _ in range(30):
                    result = post(RUNNER, '/run', dict(
                        cubin_id=compiled['cubin_id'], entry=name,
                        **{k: launch[k] for k in ('grid', 'block', 'smem_bytes', 'zero_outputs', 'args')},
                        warmup=2, iters=3, max_return=4096))
                    if result.get('stage') != 'busy':
                        break
                    time.sleep(2)
                assert result.get('ok'), result
                check_outputs({'add': expected}, result['outputs'], rtol=1e-5, atol=2e-6)
            print(f'PASS {"GPU" if remote else "compile"}: {name}', flush=True)
        except Exception as exc:
            failures.append(name)
            print(f'FAIL {name}: {exc}', flush=True)
    assert not failures, f'Failed cases: {failures}'


if __name__ == '__main__':
    if sys.argv[1:] not in ([], ['--remote']):
        sys.exit('usage: python test_edge_cases.py [--remote]')
    main(remote=bool(sys.argv[1:]))
