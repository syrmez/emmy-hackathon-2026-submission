"""Offline checks for benchmark correctness gates and safe HTML embedding."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from benchmark import DATA_TAG, check_outputs, render_snapshot

ROOT = Path(__file__).resolve().parent


def main():
    expected = {"out": np.array([1, 2], dtype=np.float32)}
    good = {"out": {"values": [1, 2], "truncated": False}}
    assert check_outputs(expected, good, 0, 0) == 0
    near = {"out": {"values": [1, 2.000001]}}
    assert 0 < check_outputs(expected, near, 1e-5, 0) < 1e-5
    for bad in ({}, {"out": {"values": [1, 2], "truncated": True}},
                {"out": {"values": [1]}}, {"out": {"values": [1, 3]}},
                {"out": {"values": [1, float("nan")]}},
                {"out": {"values": [1, float("inf")]}}):
        try:
            check_outputs(expected, bad, 0, 0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad output was accepted: {bad}")
    try:
        check_outputs({}, {}, 0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("empty reference must not count as a pass")

    template = (ROOT / "performance.html").read_text()
    report = {"rows": [], "diagnostic": "</script><script>alert('not executable')</script>"}
    rendered = render_snapshot(template, report)
    data = rendered.split(DATA_TAG, 1)[1].split("</script>", 1)[0]
    assert json.loads(data) == report
    assert "<script>" not in data and "</script>" not in data
    assert render_snapshot(rendered, report) == rendered, "refresh must only replace the data"
    try:
        render_snapshot(template, {"latency": float("nan")})
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite timing cannot be serialized")

    result = subprocess.run([sys.executable, str(ROOT / "benchmark.py"), "--iters", "0"],
                            capture_output=True, text=True)
    assert result.returncode == 2 and "iters=1..200" in result.stderr
    print("PASS: correctness gate, malformed/truncated outputs, safe HTML, repeatable refresh, CLI limits")


if __name__ == "__main__":
    main()
