# Tests and reproducible evidence

Run commands from the repository root after installing `requirements.txt`.

## Offline checks

```sh
python tests/test_compile.py          # 15 public + 24 generated problems, ABI, launch bounds, mutations
python tests/test_edge_cases.py       # 25 semantic adversaries; independent NumPy references
python tests/test_benchmark.py        # numerical checker and report safeguards
python tests/test_mnist.py            # assets, chaining, cache and HTTP boundaries
python tests/tools/benchmark_vector_loads.py
python tests/tools/update_softmax_walkthrough.py --check
node tests/test_softmax.mjs
node tests/test_softmax_browser.mjs
```

Browser checks require Node 22+ and Chrome; set `CHROME` if it is not installed at the default macOS path. The softmax tests check a JavaScript simulation, not GPU results.

## GPU numerical checks

```sh
python tests/test_compile.py --remote
python tests/test_edge_cases.py --remote
python tests/test_mnist.py --remote   # additionally needs PyTorch and MNIST downloads/cache
python mnist/server.py               # separate terminal for the browser check below
node tests/test_mnist_browser.mjs
```

The runner assembles our PTX and executes it. Expected arrays come from independent NumPy calculations; MNIST activations, logits and probabilities are compared with PyTorch. No CPU prediction fallback is used by the demo. The browser check performs real GPU inference as well as controlled transport-failure tests.

## Fixtures

- `problems/`: 15 organizer public cases; original provenance retained.
- `local_problems/`: 24 generated development cases, now included so a fresh checkout runs the full suite. Each includes IR, launch metadata/seeds/tolerances, NumPy expected arrays and generated reference CUDA. The compiler consumes only IR and launch metadata, never reference CUDA or expected arrays.
- `tools/expressions_local.py`: original expressions and independent NumPy reference formulas; covers tails, broadcasting, reduction schedules, fused operations, rectangular/tail matrix multiplication and attention.
- `test_edge_cases.py`: constructs another 25 semantic cases in memory; no generated files need uploading.

Regeneration is optional and needs the Emmy tracing dependencies (including PyTorch); running the saved fixtures does not. Target defaults to `sm_89`. To avoid overwriting the historical fixtures, regenerate elsewhere:

```sh
python tests/tools/generate.py tests/tools/expressions_local.py \
  --emmy .venv/bin/emmy --out /tmp/emmy-local-problems --target sm_89
```

Fixtures demonstrate reproducibility and coverage, **not that a run occurred**. They are development cases used during debugging, not an untouched holdout or evidence of hidden-test performance.

## Saved execution evidence

- `results/cleanup-verification.log`: fresh post-reorganization GPU run of all 39 fixtures, the renamed/reordered mutation and 25 semantic adversaries; commands, exit statuses, timestamps, environment and compiler SHA-256 recorded.
- `results/submission-performance.json`: public GPU correctness and timing snapshot, with timestamp, environment and compiler SHA-256; summarized in [RESULTS.md](RESULTS.md).
- `results/vector-loads.json`: synthetic scalar/vector comparison, not a submission-wide speedup claim.
- `performance.html`: older embedded public/generated timing snapshot; retains its own timestamp and compiler fingerprint.
- `results/compiler-gpu-regression.log`: preserved local `/tmp/emmy-hidden-readiness-after.log`, showing 39 problems plus the renamed/reordered mutation passing on GPU. The original log has no timestamp or compiler hash; it is historical supporting evidence, not a newly authenticated run. Its filename referred to readiness, **not execution of organizer hidden tests**.
- `results/mnist-gpu-regression.log`: preserved local MNIST run transcript (24 GPU inputs).
- `../mnist/assets/verification.json`: structured MNIST verification, bound to compiler, weights and pipeline hashes. Kept with demo assets because the page reads it at runtime.

No historical result files have been regenerated or relabeled as current. A test script alone is not a saved execution result; rerun the commands to verify the current checkout. For a new transcript, redirect stdout/stderr and record the command, exit status, timestamp, environment and compiler SHA-256 alongside it.

To refresh the performance dashboard separately:

```sh
python tests/benchmark.py --include-local --output /tmp/emmy-performance.html
python tests/tools/benchmark_vector_loads.py --remote --output /tmp/emmy-vector-loads.json
```
