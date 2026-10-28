"""Loopback-only MNIST demo. Inference executes our PTX on the hosted GPU, never on CPU."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlsplit

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compile import compile_file
RUNNER = "http://runner.riftstack.ai:48701"

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"


def validate_pixels(payload):
    pixels = payload.get("pixels") if isinstance(payload, dict) else None
    if not isinstance(pixels, list) or len(pixels) != 784:
        raise ValueError("Expected 784 grayscale pixels in a flat list.")
    if any(type(v) not in (int, float) or not 0 <= v <= 1 or not math.isfinite(v) for v in pixels):
        raise ValueError("Pixels must be finite numbers between 0 and 1.")
    if not any(pixels):
        raise ValueError("Draw a digit first; the canvas is blank.")
    return pixels


def output_values(result, arg):
    outputs = result.get("outputs")
    output = outputs.get(arg["name"]) if isinstance(outputs, dict) else None
    if not isinstance(output, dict):
        raise RuntimeError(f"Runner omitted output: {arg['name']}")
    values = output.get("values")
    if (output.get("truncated") or not isinstance(values, list)
            or len(values) != math.prod(arg["shape"])
            or output.get("shape") != arg["shape"] or output.get("dtype") != "f32"
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
        raise RuntimeError(f"Runner returned invalid/truncated output: {arg['name']}")
    return values


class Demo:
    def __init__(self, assets=ASSETS, runner=RUNNER):
        self.assets, self.runner = Path(assets), runner.rstrip("/")
        self.info = json.loads((self.assets / "model.json").read_text())
        weights_file = self.assets / "weights.npz"
        if hashlib.sha256(weights_file.read_bytes()).hexdigest() != self.info["weights_sha256"]:
            raise ValueError("Model weights do not match model.json; rebuild the assets.")
        with np.load(weights_file, allow_pickle=False) as weights:
            shapes = dict(w1=(784, 128), b1=(128,), w2=(128, 10), b2=(10,))
            for name, shape in shapes.items():
                if weights[name].shape != shape or weights[name].dtype != np.float32 or not np.isfinite(weights[name]).all():
                    raise ValueError(f"Invalid model tensor: {name}")
            self.weights = {name: weights[name].ravel().tolist() for name in shapes}
        self.compiler_sha256 = hashlib.sha256((ROOT / "compile.py").read_bytes()).hexdigest()
        self.kernels = []
        for stage in self.info["stages"]:
            folder = self.assets / stage["id"]
            meta = json.loads((folder / "meta.json").read_text())
            if set(meta["graph_inputs"]) != set(stage["bindings"]):
                raise ValueError(f"Model bindings do not match Emmy inputs: {stage['id']}")
            for launch in meta["kernels"]:
                started = time.perf_counter()
                ptx = compile_file(folder / f"{launch['entry']}.ir.json")
                compile_ms = (time.perf_counter() - started) * 1000
                self.kernels.append(dict(
                    **launch, index=len(self.kernels), stage=stage["id"], title=stage["title"],
                    source=stage["source"], description=stage["description"], bindings=stage["bindings"],
                    graph_output=meta["graph_outputs"][0], emmy_expression=meta["code"],
                    ir=(folder / f"{launch['entry']}.ir.txt").read_text(), ptx=ptx,
                    ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(), local_compile_ms=compile_ms))
        self.pipeline_sha256 = hashlib.sha256(json.dumps(
            [{k: v for k, v in kernel.items() if k != "local_compile_ms"} for kernel in self.kernels],
            sort_keys=True).encode()).hexdigest()
        self.cubins = {}
        self.session = requests.Session()
        # ponytail: one interactive inference at a time; use a job queue for a multi-user service.
        self.lock = threading.Lock()

    def manifest(self):
        verification = self.assets / "verification.json"
        checked = json.loads(verification.read_text()) if verification.exists() else None
        if checked and (checked.get("compiler_sha256") != self.compiler_sha256
                        or checked.get("weights_sha256") != self.info["weights_sha256"]
                        or checked.get("pipeline_sha256") != self.pipeline_sha256):
            checked = None
        return dict(**self.info, kernels=self.kernels, runner=self.runner,
                    compiler_sha256=self.compiler_sha256, pipeline_sha256=self.pipeline_sha256, verification=checked)

    def health(self):
        response = requests.get(f"{self.runner}/health", timeout=(5, 8))
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict) or not result.get("ok") or not result.get("can_run") or not result.get("can_compile"):
            raise RuntimeError("The hosted runner is not ready to compile and run.")
        return result

    def post(self, route, body):
        response = self.session.post(f"{self.runner}{route}", json=body, timeout=(5, 30))
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise RuntimeError("Malformed response from the GPU runner.")
        return result

    def assemble(self, kernel, emit):
        index = kernel["index"]
        if index not in self.cubins:
            emit(dict(type="compiling", index=index))
            result = self.post("/compile", {"ptx": kernel["ptx"]})
            if not result.get("ok") or kernel["entry"] not in result.get("entry_points", []):
                raise RuntimeError(f"ptxas rejected {kernel['entry']}: {result.get('stderr', 'missing entry point')}")
            if not isinstance(result.get("cubin_id"), str) or not result["cubin_id"]:
                raise RuntimeError("Runner did not return a cubin handle.")
            self.cubins[index] = result
        return self.cubins[index]

    def predict(self, pixels, emit=lambda event: None):
        validate_pixels({"pixels": pixels})
        started = time.perf_counter()
        tensors = dict(self.weights, pixels=pixels)
        produced, runs = {}, []
        for kernel in self.kernels:
            index = kernel["index"]
            args = []
            for argument in kernel["args"]:
                arg = {k: v for k, v in argument.items() if k not in ("seed", "intermediate")}
                if not arg.get("output") and "data" not in arg:
                    name = arg["name"]
                    arg["data"] = (produced[name] if argument.get("intermediate")
                                   else tensors[kernel["bindings"][name]])
                    if len(arg["data"]) != math.prod(arg["shape"]):
                        raise RuntimeError(f"Tensor size mismatch: {name}")
                args.append(arg)
            # A retired/restarted runner can lose its cached cubin. Retry only a lookup failure.
            for attempt in range(2):
                compiled = self.assemble(kernel, emit)
                emit(dict(type="launch", index=index, registers=compiled.get("registers", {}).get(kernel["entry"])))
                sent = time.perf_counter()
                body = {k: kernel[k] for k in ("entry", "grid", "block", "smem_bytes", "zero_outputs")}
                body.update(cubin_id=compiled["cubin_id"], args=args, warmup=0, iters=1,
                            max_return=sum(math.prod(a["shape"]) for a in args if a.get("output")))
                for retry in range(4):
                    result = self.post("/run", body)
                    if result.get("stage") != "busy":
                        break
                    emit(dict(type="waiting", index=index))
                    if retry < 3:
                        time.sleep(0.5 * (retry + 1))
                if result.get("stage") == "lookup" and attempt == 0:
                    self.cubins.pop(index, None)
                    continue
                break
            if not result.get("ok"):
                raise RuntimeError(f"GPU {result.get('stage', 'run')} failed: {result.get('stderr', 'runner unavailable')}")
            gpu_us = result.get("median_us")
            if type(gpu_us) not in (int, float) or not math.isfinite(gpu_us) or gpu_us < 0:
                raise RuntimeError("Runner returned an invalid GPU timing.")
            outputs = {arg["name"]: output_values(result, arg) for arg in args if arg.get("output")}
            produced.update(outputs)
            if kernel["final"]:
                tensors[kernel["stage"]] = produced[kernel["graph_output"]]
            event = dict(type="kernel", index=index, gpu_us=gpu_us,
                         round_trip_ms=(time.perf_counter() - sent) * 1000, outputs=outputs,
                         registers=compiled.get("registers", {}).get(kernel["entry"]))
            runs.append(event)
            emit(event)
        probabilities = tensors["softmax"]
        if (len(probabilities) != 10 or any(p < 0 or p > 1.00001 for p in probabilities)
                or not math.isclose(sum(probabilities), 1, abs_tol=1e-4)):
            raise RuntimeError("GPU returned invalid probabilities; no prediction displayed.")
        result = dict(type="result", prediction=max(range(10), key=probabilities.__getitem__),
                      probabilities=probabilities, hidden=tensors["hidden"], logits=tensors["logits"],
                      gpu_us=sum(run["gpu_us"] for run in runs),
                      server_ms=(time.perf_counter() - started) * 1000)
        emit(result)
        return result


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(20)

    def allowed(self):
        host = self.headers.get("Host", "")
        port = self.server.server_address[1]
        return (host in (f"127.0.0.1:{port}", f"localhost:{port}")
                and self.headers.get("Origin", f"http://{host}") == f"http://{host}")

    def respond(self, body, content_type="application/json", status=200):
        if not isinstance(body, bytes):
            body = json.dumps(body, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.allowed():
            return self.respond({"error": "Only same-origin loopback access is allowed."}, status=403)
        path = urlsplit(self.path).path
        if path == "/":
            return self.respond((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/model":
            return self.respond(self.server.demo.manifest())
        if path == "/api/ptx":
            ptx = "\n".join(f"// {k['stage']}: {k['entry']}\n{k['ptx']}" for k in self.server.demo.kernels)
            return self.respond(ptx.encode(), "text/plain; charset=utf-8")
        if path == "/api/health":
            try:
                return self.respond(self.server.demo.health())
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                return self.respond({"error": f"Hosted runner unavailable: {exc}"}, status=503)
        return self.respond({"error": "Not found"}, status=404)

    def do_POST(self):
        if not self.allowed():
            return self.respond({"error": "Only same-origin loopback access is allowed."}, status=403)
        if self.path != "/api/predict":
            return self.respond({"error": "Not found"}, status=404)
        if self.headers.get_content_type() != "application/json" or self.headers.get("Transfer-Encoding"):
            return self.respond({"error": "Send a bounded JSON request."}, status=415)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("Request body must be 1..65536 bytes.")
            pixels = validate_pixels(json.loads(self.rfile.read(length)))
        except (ValueError, OverflowError, RecursionError) as exc:
            return self.respond({"error": str(exc)}, status=400)
        demo = self.server.demo
        if not demo.lock.acquire(blocking=False):
            return self.respond({"error": "Another inference is running. Try again in a moment."}, status=409)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()

            def emit(event):
                self.wfile.write((json.dumps(event, allow_nan=False) + "\n").encode())
                self.wfile.flush()

            try:
                demo.predict(pixels, emit)
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                emit(dict(type="error", message=str(exc)))
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass  # Browser left; do not launch the remaining kernels.
        finally:
            demo.lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--runner", default=RUNNER)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")
    try:
        demo = Demo(runner=args.runner)
    except FileNotFoundError:
        parser.error("Missing model assets. Run .venv/bin/python mnist/prepare.py first.")
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        parser.exit(1, f"Cannot listen on port {args.port}: {exc}. Try --port <another-port>.\n")
    server.demo = demo
    print(f"MNIST compiler demo: http://127.0.0.1:{args.port}", flush=True)
    print(f"{len(demo.kernels)} kernels emitted by compile.py; GPU execution: {demo.runner}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        demo.session.close()


if __name__ == "__main__":
    main()
