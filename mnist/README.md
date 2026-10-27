# From ink to instructions

A local browser demo of **our Kernel IR → PTX compiler**, with actual MNIST inference on the hackathon's hosted RTX 5090. No CPU inference fallback, frontend framework, npm install, local CUDA, or local GPU required.

## Run

From the repository root, using the existing Python environment:

```bash
.venv/bin/python mnist/server.py
open http://127.0.0.1:8766
```

A fresh environment needs only the root `requirements.txt` (Python 3.12+). The trained weights and Emmy IR are included in `mnist/assets/`; **PyTorch is not needed to serve the demo**. The server binds to loopback only. Use `--port 8767` if the port is occupied; stop with Ctrl-C.

- Draw with mouse, touch or pen; prediction starts after a stroke. Multiple strokes during an inference coalesce to the newest drawing. Clearing the canvas invalidates an in-flight result.
- Try the 0–9 sample buttons for keyboard-accessible input. These are the first official test-set example of each digit, not examples selected for correct predictions.
- Switch among PyTorch, actual Kernel IR and actual generated PTX. Every kernel's complete PTX is available; the download bundles separate PTX modules as text, not one assemblable module.
- Enable **Step-through mode** to inspect the last inference's actual intermediate tensors, navigate kernels and highlight instruction categories. It is optional and off by default.

The input preview is exactly the 784 values sent to the model. Drawings are cropped, fitted inside 20×20, centered by intensity centroid in a 28×28 frame, and scaled to [0, 1]. Original MNIST samples bypass this extra cropping. The demo does not persist drawings; pixels and public model weights are sent through the local server to the hackathon runner, whose provided endpoint uses HTTP. This is a demo endpoint, not a private production service.

## What really executes

The trained float32 MLP has **101,770 parameters**, with weights stored pre-transposed for the traced matrix multiplications:

```python
# pixels: [1, 784], W1: [784, 128], W2: [128, 10]
hidden = torch.relu(pixels @ W1 + b1)
logits = hidden @ W2 + b2
probabilities = torch.softmax(logits, dim=-1)
```

`prepare.py` uses the organizer's existing `tools/generate.py` to obtain Emmy IR and launch descriptors, preserving three stage boundaries for inspection. The generator's random inputs only specify tensor shapes; inference supplies the actual pixels and learned weights. The generator may produce reference CUDA, but it is discarded and **never used by our compiler or the inference path**.

`server.py` invokes the unchanged root `compile.py` on that IR at startup. Emmy schedules four kernels:

1. **Partial dot products** — split each 784-term dot product into eight chunks of 98; return 8×128 partial sums.
2. **Bias + ReLU** — combine partial sums, add bias, apply ReLU; return 128 features.
3. **Digit scores** — weighted combination and bias; return ten logits.
4. **Softmax** — return ten probabilities.

Our emitted PTX is assembled by the hosted runner's `ptxas`, then executed with the exact positional arguments, launch geometry and output-zeroing rules in `meta.json`. Intermediate buffers returned by one GPU request become the next request's inputs. The browser's displayed digit is the argmax of the GPU's probabilities; no CPU neural-network operations produce the prediction.

PTX is **static generated code**, not a dynamic instruction trace. Binaries are assembled lazily and cached; changing a drawing changes data, not instructions. The live strip records HTTP submission/completion events, not exact GPU start times. The educational view replays completed kernel outputs, not single instructions. `ptxas` ultimately assembles PTX into machine code (SASS).

## Checks and measurements

```bash
.venv/bin/python mnist/test_demo.py           # offline: data validation, chaining, cache, HTTP safety
.venv/bin/python mnist/test_demo.py --remote  # actual GPU vs PyTorch; needs torch + internet
node mnist/test_browser.mjs                  # server must be running; Node 22+ and Chrome
```

The browser check uses Chrome's DevTools protocol and Node's built-in WebSocket, with no testing packages. Override `CHROME` and `DEMO_URL` if necessary. It tests real sample and pointer-drawn inference, code views, optional steps, preprocessing, stale-result handling, failure states and mobile layout; screenshots go into the OS temporary directory. Synthetic transport responses are confined to the browser test's failure/stale-result checks and never used by the application.

Recorded results in `assets/model.json` and `assets/verification.json`:

- **98.16%** (9,816/10,000) on the official MNIST test set, evaluated in **PyTorch on CPU** after a fixed training budget. This is not a guarantee on arbitrary mouse handwriting and not a full-test-set GPU accuracy claim.
- **24 GPU inputs checked**: the first 20 test images plus the first example of every class (21 distinct images), and three synthetic stress inputs. All 128 hidden activations, ten logits and ten probabilities match PyTorch at `rtol=2e-4, atol=2e-5`; all top-1 predictions agree.
- Maximum absolute probability error: **1.79e-7**. All 21 sampled MNIST labels were recognized correctly; this small development check is not another accuracy estimate.

Verification is bound to the compiler, weights and generated pipeline hashes; a changed pipeline does not display the old check as current. GPU timings sum four single-launch measurements with no warmup and no clock locking: useful feedback, **not a benchmark or a speedup claim**. End-to-end time includes JSON encoding, uploads, queueing, GPU copies, HTTP round trips and first-use assembly. In this session, warmed requests were roughly 2–3 seconds despite microsecond-scale GPU work.

Deliberate demo limits: one in-flight inference per server, fixed batch size 1 and fixed model, public runner availability, and HTTP transfers between kernels. The runner API does not retain model buffers across these calls, so weights are re-uploaded. A deployment needing low latency would use a persistent GPU runtime rather than this test runner. Runner failures are shown explicitly; there is no fabricated prediction or silent CPU fallback.

## Rebuild the model and IR

```bash
uv pip install --python .venv/bin/python torch==2.14.1
# Move mnist/assets aside first; preparation refuses to overwrite existing assets.
.venv/bin/python mnist/prepare.py --epochs 12
.venv/bin/python mnist/test_demo.py --remote
```

Training uses seed 42, CPU Adam, 60,000 training images, batch size 256, and train-only rotation/scale/translation augmentation. Official gzipped MNIST IDX files are downloaded over HTTPS from the OSSCI mirror, checked against their published checksums, and cached under `~/.cache/emmy-mnist`. No torchvision or executable downloaded checkpoint is used. The test set is evaluated after training, not used to select epochs. Exact training results can vary with PyTorch/platform versions.
