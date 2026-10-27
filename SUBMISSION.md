# Compiler submission

A direct Emmy IR-to-PTX compiler. The lowering is implemented in `compile.py`; it does not call an existing compiler backend or read reference outputs.

```sh
python -m pip install -r requirements.txt
python compile.py problems/p00_scalar_add/k_add_pointwise.ir.json > out.ptx
```

Requires Python 3.12+. Keep the input IR beside its `meta.json` (argument order and launch configuration). Stdout contains PTX only.

## Public results

**15/15 public problems passed full-array correctness checks at their original tolerances.** Median compiler CLI latency across problems: **130.85 ms**; slowest: **157.40 ms**.

| Public problem | Correctness | Compiler CLI (ms) | GPU kernel (µs) |
|---|---|---:|---:|
| a01_attention | pass | 136.46 | 98.848 |
| a02_attention_128 | pass | 130.85 | 68.128 |
| f01_rmsnorm | pass | 128.51 | 5.184 |
| f02_row_softmax | pass | 148.45 | 3.072 |
| f03_l2norm | pass | 131.31 | 3.040 |
| m01_matmul_512 | pass | 137.75 | 152.064 |
| m02_matmul_rect | pass | 130.47 | 4.864 |
| p00_scalar_add | pass | 132.20 | 3.040 |
| p01_vector_add | pass | 128.14 | 3.056 |
| p02_scale_shift | pass | 128.21 | 3.104 |
| p03_relu_mul | pass | 127.34 | 3.072 |
| r01_array_sum | pass | 157.40 | 3.072 |
| r02_big_sum | pass | 128.43 | 21.024 |
| r03_mean | pass | 131.64 | 12.832 |
| r04_row_sum | pass | 128.31 | 3.040 |

Snapshot: 2026-10-27, NVIDIA RTX 5090 (`sm_120`); compiler timing on macOS 14.5 ARM64, Python 3.12.4. Each compiler result is the median of three complete CLI invocations, including Python startup/imports but excluding remote assembly and HTTP. GPU results are medians of three batch medians, each with 10 warmups and 100 timed iterations. Each batch's final output is checked; atomic outputs are reset before each launch. Shared GPU and unlocked clocks limit timing generalization; no baseline speedup is claimed.

Raw measurements: [`benchmarks/submission-performance.json`](benchmarks/submission-performance.json).
Compiler SHA-256: `de93a1a6260fc420ad409fa3c618ba03d99c74cf8c6ed256a1d995cbfb5511ce`.

## Validation

```sh
python test_compile.py                 # public/generated IR, ABI, alignment, launch validation
python test_edge_cases.py              # 25 focused IR semantic cases
python test_benchmark.py               # correctness checker and report safeguards
python test_compile.py --remote        # assemble/run on the hosted GPU
python test_edge_cases.py --remote     # compare GPU results against NumPy
```

The recorded GPU regression suite at commit `52282cb` passed 15 public problems, 24 generated problems, one renamed/reordered-argument mutation, and 25 semantic edge cases. Generated problems are git-ignored; their definitions are committed in `tools/expressions_local.py`. Generate them with the organizer's `tools/generate.py` to include them in `test_compile.py`.

Tests cover tails, broadcasting, reduction schedules, typed predicates, scope, loops, warp reductions, async copies, atomics, vector alignment and metadata independence. These are development/regression tests, **not a hidden-set score or untouched holdout**. Float32-only computation and unsupported operations/layouts/schedules remain limitations; unsupported inputs fail explicitly rather than silently producing an approximate substitute.
