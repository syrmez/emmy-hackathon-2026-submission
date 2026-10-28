"""Run offline checks; --remote also checks real GPU outputs against PyTorch."""
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mnist"))
from server import ASSETS, Demo, Handler, output_values, validate_pixels


def rejects(function, exception=ValueError):
    try:
        function()
    except exception:
        return
    raise AssertionError("Invalid input/result was accepted")


def offline():
    pixels = [0.5] * 784
    assert validate_pixels({"pixels": pixels}) == pixels
    for bad in (None, [], {}, {"pixels": []}, {"pixels": [0] * 784}, {"pixels": [0.1] * 783}):
        rejects(lambda: validate_pixels(bad))
    for value in (True, "0.1", None, [], -0.1, 1.1, float("nan"), float("inf"), 10**400):
        rejects(lambda: validate_pixels({"pixels": [value, *pixels[1:]]}))
    arg = dict(name="out", shape=[1, 2])
    valid = dict(values=[1, 2], shape=[1, 2], dtype="f32", truncated=False)
    assert output_values({"outputs": {"out": valid}}, arg) == [1, 2]
    for bad in (None, {}, dict(valid, truncated=True), dict(valid, values=[1]),
                dict(valid, values=[1, float("nan")]), dict(valid, values=[1, True]),
                dict(valid, shape=[2]), dict(valid, dtype="f16")):
        rejects(lambda: output_values({"outputs": {"out": bad}}, arg), RuntimeError)

    demo = Demo()
    assert len(demo.kernels) == 4 and demo.info["architecture"] == [784, 128, 10]
    assert demo.pipeline_sha256 == Demo().pipeline_sha256, "Compile timings must not affect fingerprint"
    assert all(k["ptx"].startswith(".version 8.0") and k["ir"] for k in demo.kernels)
    calls, emitted = [], []
    lookup = True

    def fake_post(route, body):
        nonlocal lookup
        calls.append((route, body))
        if route == "/compile":
            entry = next(k["entry"] for k in demo.kernels if k["ptx"] == body["ptx"])
            return dict(ok=True, cubin_id=entry, entry_points=[entry], registers={entry: 20})
        if lookup:
            lookup = False
            return dict(ok=False, stage="lookup", stderr="expired cubin")
        k = next(k for k in demo.kernels if k["entry"] == body["entry"])
        assert [a["name"] for a in body["args"]] == k["arg_order"]
        assert body["zero_outputs"] == k["zero_outputs"] and body["iters"] == 1
        for a in body["args"]:
            if a.get("output"):
                continue
            assert "seed" not in a and "intermediate" not in a
            binding = k["bindings"].get(a["name"])
            expected = pixels if binding == "pixels" else demo.weights.get(binding, [0.25] * len(a["data"]))
            assert a["data"] == expected, "Must chain GPU results and use the trained weights"
        return dict(ok=True, median_us=1, outputs={
            a["name"]: dict(values=[0.1 if k["stage"] == "softmax" else 0.25] * int(np.prod(a["shape"])),
                            shape=a["shape"], dtype="f32", truncated=False)
            for a in body["args"] if a.get("output")})

    demo.post = fake_post
    result = demo.predict(pixels, emitted.append)
    assert result["probabilities"] == [0.1] * 10 and result["gpu_us"] == 4
    assert sum(route == "/compile" for route, _ in calls) == 5, "Only expired lookup is reassembled"
    assert [e["index"] for e in emitted if e["type"] == "kernel"] == [0, 1, 2, 3]
    demo.predict(pixels)
    assert sum(route == "/compile" for route, _ in calls) == 5, "Reuse cached binaries"
    demo.post = lambda route, body: dict(ok=False, stage="run", stderr="test failure")
    failures = []
    rejects(lambda: demo.predict(pixels, failures.append), RuntimeError)
    assert not any(e["type"] == "result" for e in failures), "Never substitute a CPU prediction"

    # Exercise actual HTTP boundaries without contacting the GPU.
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.demo = demo
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert requests.get(url + "/api/model", timeout=5).json()["pipeline_sha256"] == demo.pipeline_sha256
        assert requests.get(url + "/api/ptx", timeout=5).text.count(".version 8.0") == 4
        assert requests.get(url + "/../compile.py", timeout=5).status_code == 404
        assert requests.get(url + "/api/model", headers={"Host": "evil.example"}, timeout=5).status_code == 403
        assert requests.post(url + "/api/predict", json={"pixels": pixels}, headers={"Origin": "https://evil.example"}, timeout=5).status_code == 403
        assert requests.post(url + "/api/predict", json={"pixels": []}, timeout=5).status_code == 400
        assert requests.post(url + "/api/predict", data="x" * 70000, headers={"Content-Type": "application/json"}, timeout=5).status_code == 400
        assert requests.post(url + "/api/predict", data="hello", timeout=5).status_code == 415
        demo.lock.acquire()
        try:
            assert requests.post(url + "/api/predict", json={"pixels": pixels}, timeout=5).status_code == 409
        finally:
            demo.lock.release()
        events = [json.loads(line) for line in requests.post(url + "/api/predict", json={"pixels": pixels}, timeout=5).text.splitlines()]
        assert events[-1]["type"] == "error" and "test failure" in events[-1]["message"]
        assert demo.lock.acquire(blocking=False), "Release lock after a failed stream"
        demo.lock.release()
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
        demo.session.close()
    print("PASS: assets, input/output validation, positional ABI, GPU chaining, caching, expired handles, failed inference, HTTP boundaries", flush=True)


def remote():
    import torch
    from prepare import dataset

    torch.set_num_threads(4)
    demo = Demo()
    health = demo.health()
    images, labels = dataset("t10k")
    # Fixed first 20 images + first example of every digit, without selecting for correctness.
    indices = sorted(set(range(20)) | {s["test_index"] for s in demo.info["samples"]})
    cases = [(f"MNIST test #{i}", images[i].ravel().astype(np.float32) / 255, int(labels[i])) for i in indices]
    rng = np.random.default_rng(42)
    cases += [("all white", np.ones(784, np.float32), None),
              ("noise", rng.random(784).astype(np.float32), None),
              ("single pixel", np.eye(1, 784, 390, dtype=np.float32).ravel(), None)]
    with np.load(ASSETS / "weights.npz", allow_pickle=False) as archive:
        w = {name: torch.from_numpy(archive[name]) for name in archive.files}
    errors = dict(hidden=0.0, logits=0.0, probabilities=0.0)
    correct = 0
    try:
        for name, pixels, label in cases:
            x = torch.from_numpy(pixels).reshape(1, 784)
            with torch.inference_mode():
                hidden = torch.relu(x @ w["w1"] + w["b1"])
                logits = hidden @ w["w2"] + w["b2"]
                probabilities = torch.softmax(logits, dim=-1)
            events = []
            result = demo.predict(pixels.tolist(), events.append)
            for key, expected in (("hidden", hidden), ("logits", logits), ("probabilities", probabilities)):
                reference = expected.numpy().ravel()
                np.testing.assert_allclose(result[key], reference, rtol=2e-4, atol=2e-5, err_msg=f"{name}: {key}")
                errors[key] = max(errors[key], float(np.max(np.abs(np.array(result[key]) - reference))))
            assert result["prediction"] == int(probabilities.argmax())
            assert len([e for e in events if e["type"] == "kernel"]) == len(demo.kernels)
            correct += label is not None and result["prediction"] == label
            print(f"PASS: {name}: {result['prediction']} (label {label}), {result['server_ms'] / 1000:.2f}s end-to-end", flush=True)
    finally:
        demo.session.close()
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(), cases=len(cases), mnist_indices=indices,
                  mnist_correct=correct, mnist_count=len(indices), synthetic_count=3,
                  max_abs_error=errors, rtol=2e-4, atol=2e-5,
                  compiler_sha256=demo.compiler_sha256, weights_sha256=demo.info["weights_sha256"],
                  pipeline_sha256=demo.pipeline_sha256, gpu=health.get("gpu"), runner=demo.runner)
    (ASSETS / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(cases)} real GPU inputs; all activations/logits/probabilities match PyTorch. Errors: {errors}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["--remote"]):
        sys.exit("usage: python tests/test_mnist.py [--remote]")
    offline()
    if "--remote" in sys.argv:
        remote()
