"""Run offline checks, or add GPU correctness checks with --remote.

Generate local_problems/ first to include the extra development cases.
The mutation test uses only IR + meta + an independent NumPy expected output.
"""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from compile import compile_file
from emmy.compiler.graph import Graph
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var

ROOT = Path(__file__).resolve().parent


def check_remote(folder):
    subprocess.run([sys.executable, str(ROOT / "submit.py"), str(folder),
                    "--compiler", str(ROOT / "compile.py")], check=True)


def main(remote=False):
    folders = sorted(p.parent for p in (ROOT / "problems").glob("*/meta.json"))
    folders += sorted((ROOT / "local_problems").glob("v*"))
    for folder in folders:
        for ir in sorted(folder.glob("*.ir.json")):
            ptx = compile_file(ir)
            assert ptx.isascii() and ptx.startswith(".version 8.0\n")
            assert "setp.ge.u32" in ptx
            assert "st.global.f32" in ptx or "red.global.add.f32" in ptx
        if remote:
            check_remote(folder)
    print(f"PASS: compiled kernels for {len(folders)} problems", flush=True)

    # No filename/entry special cases, no reference files, and positional args
    # deliberately reversed. Also exercise grouped loads and signed floor/mod.
    source = ROOT / "problems/p00_scalar_add"
    graph = Graph.from_dict(json.loads(next(source.glob("*.ir.json")).read_text()))
    node = graph.nodes["add"]
    tile = node.op.body[0]
    s = tile.body
    a = Var(tile.axes[0].name)
    two = Literal(2, "int")
    x_index = (BinaryExpr("//", a * two - Literal(7, "int"), two) + Literal(4, "int")) * two
    y_index = ((a + Literal(1, "int")) % Literal(-64, "int") + Literal(63, "int")) * two
    body = (
        replace(s[0], names=(s[0].name, s[3].name), index=(x_index,)),
        replace(s[1], names=(s[1].name, s[4].name), index=(y_index,)),
        s[2], s[5], s[6],
    )
    name = "unseen_grouped_load"
    graph.nodes[node.id] = replace(node, op=replace(node.op, name=name,
                                  body=(replace(tile, body=body),)))
    meta = json.loads((source / "meta.json").read_text())
    meta.pop("code")
    k = meta["kernels"][0]
    k["entry"] = name
    k["arg_order"].reverse()
    k["args"].reverse()
    arrays = {}
    for arg in k["args"]:
        if not arg.get("output"):
            arg["seed"] += 37
            arrays[arg["name"]] = np.random.default_rng(arg["seed"]).standard_normal(arg["shape"]).astype(np.float32)

    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        ir = folder / f"{name}.ir.json"
        ir.write_text(json.dumps(graph.to_dict(), default=str))
        (folder / "meta.json").write_text(json.dumps(meta))
        ptx = compile_file(ir)
        assert f".entry {name}(" in ptx
        # Actual first buffer loaded is x0, now third in the signature: %rd5.
        first_address = next(line for line in ptx.splitlines() if "add.u64" in line)
        assert ", %rd5," in first_address
        if remote:
            np.savez(folder / "expected.npz", add=arrays["x0"] + arrays["x1"])
            check_remote(folder)
        meta["code"] = "not valid torch code"
        meta["id"] = "different_problem_label"
        for arg in k["args"]:
            if "seed" in arg:
                arg["seed"] += 1
        (folder / "meta.json").write_text(json.dumps(meta))
        assert compile_file(ir) == ptx, "compiler must ignore test seeds/labels/expression"
        # Unsupported precision must fail, never silently reinterpret as f32.
        data = graph.to_dict()
        data["nodes"]["x0"]["outputs"][0]["dtype"] = "f64"
        ir.write_text(json.dumps(data, default=str))
        result = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir)],
                                capture_output=True, text=True)
        assert result.returncode != 0 and result.stdout == "" and "f32" in result.stderr
        # Unsupported operations must also fail with no partial PTX on stdout.
        current = graph.nodes[node.id]
        bad_body = (*body[:2], replace(s[2], op=ElementwiseImpl("sin"), args=(s[0].name,)), *body[3:])
        graph.nodes[node.id] = replace(current, op=replace(current.op,
                                      body=(replace(tile, body=bad_body),)))
        ir.write_text(json.dumps(graph.to_dict(), default=str))
        result = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir)],
                                capture_output=True, text=True)
        assert result.returncode != 0 and result.stdout == "" and "sin" in result.stderr
    print("PASS: renamed entry, reordered args, grouped loads, metadata independence, clean rejections", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["--remote"]):
        sys.exit("usage: python test_compile.py [--remote]")
    main(remote=bool(sys.argv[1:]))
