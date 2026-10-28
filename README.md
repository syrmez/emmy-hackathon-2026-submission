# Emmy IR → PTX compiler

`compile.py` lowers Emmy Kernel IR directly to PTX. It does not invoke another compiler backend or read reference outputs. Requires Python 3.12+.

```sh
python -m pip install -r requirements.txt
python compile.py tests/problems/p00_scalar_add/k_add_pointwise.ir.json > out.ptx
```

Keep each input IR beside its `meta.json` (argument order and launch configuration). Stdout contains PTX only. Computation is float32; unsupported operations/layouts/schedules fail explicitly.

## Demos

- **Softmax walkthrough:** open [`softmax.html`](softmax.html) directly in a browser. This is a local educational simulation with actual IR and compiler/PTX excerpts, not live GPU execution.
- **MNIST inference:** run `python mnist/server.py`, then open http://127.0.0.1:8766. Uses the included weights and our compiled PTX on the hosted GPU; no local GPU or PyTorch needed for serving. See [`mnist/README.md`](mnist/README.md).

## Tests and evidence

[`tests/`](tests/README.md) contains compiler checks, GPU numerical comparisons, demo tests, all **15 public + 24 generated problems**, generation scripts, and saved results. See the [recorded results and limitations](tests/RESULTS.md) and [performance dashboard](tests/performance.html).

```sh
python tests/test_compile.py
python tests/test_edge_cases.py
python tests/test_benchmark.py
python tests/test_mnist.py
node tests/test_softmax.mjs
```

Add `--remote` to the compiler or edge-case checks to assemble and run on the hosted GPU. Remote checks require the organizer's runner to remain available. These are development/regression tests, not a hidden-set score or untouched holdout.
