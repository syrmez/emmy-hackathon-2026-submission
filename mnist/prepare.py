"""Train a tiny MNIST MLP and export real Emmy IR. Run once, not during inference."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile

import numpy as np
import requests
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
from generate import package, trace
from compile import compile_file

ASSETS = Path(__file__).with_name("assets")
CACHE = Path.home() / ".cache/emmy-mnist"
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def dataset(split):
    """Official IDX files; no torchvision or executable model downloads."""
    CACHE.mkdir(parents=True, exist_ok=True)
    arrays = []
    for kind in ("images-idx3", "labels-idx1"):
        name = f"{split}-{kind}-ubyte.gz"
        path = CACHE / name
        if not path.exists():
            response = requests.get(f"https://ossci-datasets.s3.amazonaws.com/mnist/{name}", timeout=(10, 120))
            response.raise_for_status()
            data = response.content
        else:
            data = path.read_bytes()
        # Published MNIST checksums detect incomplete/corrupted downloads.
        if hashlib.md5(data).hexdigest() != FILES[name]:
            raise ValueError(f"MNIST checksum mismatch: {path}")
        if not path.exists():
            path.write_bytes(data)
        raw = gzip.decompress(data)
        magic, count = struct.unpack(">II", raw[:8])
        if magic == 2051:
            assert struct.unpack(">II", raw[8:16]) == (28, 28)
            array = np.frombuffer(raw, np.uint8, offset=16).reshape(count, 1, 28, 28)
        elif magic == 2049:
            array = np.frombuffer(raw, np.uint8, offset=8).reshape(count)
        else:
            raise ValueError(f"Bad IDX header: {name}")
        arrays.append(array.copy())
    return arrays


def softmax(x):
    exp = np.exp(x - x.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


STAGES = [
    dict(id="hidden", title="Hidden layer", source="hidden = torch.relu(pixels @ W1 + b1)",
         description="784 pixels become 128 learned features. Emmy splits the dot products into partial sums, then combines them, adds the bias and applies ReLU.",
         bindings={"x": "pixels", "w": "w1", "b": "b1"},
         code="x = torch.randn(1, 784); w = torch.randn(784, 128); b = torch.randn(128); torch.relu(x @ w + b)",
         ref=lambda x, w, b: [np.maximum(x @ w + b, 0)], rtol=1e-3, atol=1e-4),
    dict(id="logits", title="Digit scores", source="logits = hidden @ W2 + b2",
         description="128 features vote for the ten digits. These raw scores (logits) can be negative and are not probabilities.",
         bindings={"h": "hidden", "w": "w2", "b": "b2"},
         code="h = torch.randn(1, 128); w = torch.randn(128, 10); b = torch.randn(10); h @ w + b",
         ref=lambda h, w, b: [h @ w + b], rtol=1e-3, atol=1e-4),
    dict(id="softmax", title="Probabilities", source="probabilities = torch.softmax(logits, dim=-1)",
         description="Subtract the maximum, exponentiate and divide by the sum. The GPU returns all ten probabilities; the browser highlights the largest.",
         bindings={"logits": "logits"},
         code="logits = torch.randn(1, 10); torch.softmax(logits, dim=-1)", ref=lambda x: [softmax(x)],
         rtol=1e-3, atol=1e-5, knobs="WORK=t32,REDUCE=coop"),
]


def build(epochs, emmy):
    torch.manual_seed(42)
    torch.set_num_threads(4)
    images, labels = dataset("train")
    x, y = torch.from_numpy(images).float() / 255, torch.from_numpy(labels).long()
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for ids in torch.randperm(len(x)).split(256):
            batch = x[ids]
            # Train-only augmentation helps with mouse/stylus handwriting.
            angle = (torch.rand(len(ids)) - 0.5) * (np.pi / 6)
            scale = 0.9 + torch.rand(len(ids)) * 0.2
            theta = torch.zeros(len(ids), 2, 3)
            theta[:, 0, 0] = theta[:, 1, 1] = angle.cos() * scale
            theta[:, 0, 1], theta[:, 1, 0] = -angle.sin() * scale, angle.sin() * scale
            theta[:, :, 2] = (torch.rand(len(ids), 2) - 0.5) * 0.14
            batch = F.grid_sample(batch, F.affine_grid(theta, batch.shape, align_corners=False), align_corners=False)
            loss = F.cross_entropy(model(batch), y[ids])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ids)
        print(f"Epoch {epoch + 1}/{epochs}: train loss {total / len(x):.4f}", flush=True)

    # Fixed training budget, then evaluate once on the unaugmented official test set.
    test_images, test_labels = dataset("t10k")
    model.eval()
    with torch.inference_mode():
        predictions = torch.cat([model(b).argmax(1) for b in (torch.from_numpy(test_images).float() / 255).split(512)])
    correct = int((predictions == torch.from_numpy(test_labels)).sum())
    print(f"MNIST test: {correct}/10000 ({correct / 100:.2f}%)", flush=True)

    # Build in a temporary directory so a failed generation keeps existing assets intact.
    ASSETS.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ASSETS.parent) as directory:
        out = Path(directory)
        weights = {"w1": model[1].weight.detach().numpy().T.copy(), "b1": model[1].bias.detach().numpy(),
                   "w2": model[3].weight.detach().numpy().T.copy(), "b2": model[3].bias.detach().numpy()}
        np.savez_compressed(out / "weights.npz", **weights)
        stages = []
        for spec in STAGES:
            with tempfile.TemporaryDirectory() as td:
                dump = Path(td) / "dump"
                trace(emmy, spec, dump, "sm_89")
                folder = out / spec["id"]
                print(package(spec, dump, folder), flush=True)
                meta = json.loads((folder / "meta.json").read_text())
                if set(meta["graph_inputs"]) != set(spec["bindings"]):
                    raise ValueError(f"Unexpected Emmy input names in {spec['id']}")
                for ir in folder.glob("*.ir.json"):
                    compile_file(ir)  # Fail here if our compiler cannot lower the actual IR.
                # Only IR + launch metadata ship. Our compiler never uses reference CUDA.
                for reference in [*folder.glob("*.ref.cu"), folder / "expected.npz"]:
                    reference.unlink()
            stages.append({k: spec[k] for k in ("id", "title", "source", "description", "bindings")})
        # First test-set example of each class, not cherry-picked by model prediction.
        samples = [{"label": digit, "test_index": int(np.flatnonzero(test_labels == digit)[0]),
                    "pixels": test_images[np.flatnonzero(test_labels == digit)[0]].ravel().tolist()}
                   for digit in range(10)]
        report = dict(created_at=datetime.now(timezone.utc).isoformat(), seed=42, epochs=epochs,
                      architecture=[784, 128, 10], parameters=sum(p.numel() for p in model.parameters()),
                      test_correct=correct, test_count=10000, training_count=60000,
                      accuracy_backend="PyTorch CPU; not a full-test-set GPU measurement",
                      weights_sha256=hashlib.sha256((out / "weights.npz").read_bytes()).hexdigest(),
                      torch_version=torch.__version__, stages=stages, samples=samples)
        (out / "model.json").write_text(json.dumps(report, indent=2) + "\n")
        if ASSETS.exists():
            raise FileExistsError(f"{ASSETS} already exists; move it aside explicitly before rebuilding")
        shutil.copytree(out, ASSETS)
    print(f"Saved {ASSETS}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--emmy", default=str(Path(sys.executable).parent / "emmy"))
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100:
        parser.error("epochs must be 1..100")
    if ASSETS.exists():
        parser.error(f"{ASSETS} already exists; move it aside explicitly before rebuilding")
    build(args.epochs, args.emmy)
