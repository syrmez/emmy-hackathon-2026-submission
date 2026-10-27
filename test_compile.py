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

from compile import Compiler, compile_file
from emmy.compiler.graph import Graph
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.stmt import Load

ROOT = Path(__file__).resolve().parent


def check_remote(folder):
    subprocess.run([sys.executable, str(ROOT / "submit.py"), str(folder),
                    "--compiler", str(ROOT / "compile.py")], check=True)


def check_vector_loads():
    graph = Graph.from_dict(json.loads(next((ROOT / "problems/p00_scalar_add").glob("*.ir.json")).read_text()))
    a = Var("i")
    # Exact widths and offsets: no over-read on tails, no vector on unknown alignment.
    cases = [
        ((), 4, [4]),
        ((a * 4,), 4, [4]),
        ((a * 2,), 4, [2, 2]),
        ((a,), 4, [1, 1, 1, 1]),
        ((a * 4 + 1,), 4, [1, 1, 1, 1]),
        ((Literal(2, "int"),), 4, [2, 2]),
        ((a * 4,), 1, [1]),
        ((a * 4,), 3, [2, 1]),
        ((a * 4,), 7, [4, 2, 1]),
        ((a * 4,), 8, [4, 4]),
        ((a, Literal(0, "int")), 4, [4]),
    ]
    for shared, alignment in ((False, 16), (True, 4), (True, 8), (True, 16)):
        for index, count, widths in cases:
            compiler = Compiler(graph, None, {"smem_bytes": 0})
            compiler.indices["i"] = "%r99"
            compiler.pointers["x0"] = "%r98" if shared else "%rd99"
            compiler.shapes["x0"] = (8, 16)
            compiler.alignments["x0"] = alignment
            if shared:
                compiler.shared["x0"] = "shared_test"
            names = tuple(f"value{j}" for j in range(count))
            compiler.statement(Load(input="x0", index=index, names=names))
            expected_widths = [min(w, alignment // 4) for w in widths
                               for _ in range(max(1, w // (alignment // 4)))]
            loads = [line.strip() for line in compiler.lines if line.strip().startswith("ld.")]
            address = loads[0].split("[")[1].split("+")[0]
            expected = []
            offset = 0
            for width in expected_widths:
                regs = [f"%f{j}" for j in range(offset, offset + width)]
                target = regs[0] if width == 1 else "{" + ", ".join(regs) + "}"
                vector = "" if width == 1 else f".v{width}"
                expected.append(f"ld.{compiler.space('x0')}{vector}.f32 {target}, [{address}+{offset * 4}];")
                offset += width
            assert loads == expected, (index, alignment, loads, expected)
            assert list(compiler.values.values()) == [f"%f{j}" for j in range(count)]
    print("PASS: vector load widths, register bindings, offsets, tails, and alignment fallbacks", flush=True)


def check_launch_ranges():
    source = ROOT / "problems/p00_scalar_add"
    ir = next(source.glob("*.ir.json"))
    graph = Graph.from_dict(json.loads(ir.read_text()))
    kernel = graph.nodes["add"].op
    launch = json.loads((source / "meta.json").read_text())["kernels"][0]
    # Compilation only: do NOT launch the enormous boundary grids on a GPU.
    cases = [(grid, block, None) for grid, block in (
        ([64, 1, 1], [1, 1, 1]),
        ([1, 1, 1], [64, 1, 1]),
        ([1, 1, 1], [1024, 1, 1]),
        ([2**31 - 1, 1, 1], [1, 1, 1]),
        ([2**31 - 1, 1, 1], [2, 1, 1]),
        ([2**22, 1, 1], [1024, 1, 1]),  # Exactly 2**32 threads.
    )]
    cases += [
        ([1, 1, 1], [1025, 1, 1], "block.x"),
        ([2**31, 1, 1], [1, 1, 1], "grid.x"),
        ([2**22 + 1, 1, 1], [1024, 1, 1], "global thread IDs"),
        ([1, 2, 1], [256, 1, 1], "1-D"),
        ([1, 1, 2], [256, 1, 1], "1-D"),
        ([1, 1, 1], [256, 2, 1], "1-D"),
        ([1, 1, 1], [256, 1, 2], "1-D"),
        ([1, 1, 1], [63, 1, 1], "does not cover"),
    ]
    for bad in (None, 1, "1,1,1", {}, [], [1], [1, 1], [1, 1, 1, 1],
                [0, 1, 1], [-1, 1, 1], [1, 0, 1], [1, 1, -1],
                [1.0, 1, 1], [True, 1, 1], ["1", 1, 1], [1, False, 1]):
        cases.append((bad, [256, 1, 1], "grid must be three positive integers"))
        cases.append(([1, 1, 1], bad, "block must be three positive integers"))
    for grid, block, error in cases:
        compiler = Compiler(graph, kernel, launch | {"grid": grid, "block": block})
        try:
            ptx = compiler.compile()
        except (ValueError, NotImplementedError) as exc:
            assert error and error in str(exc), (grid, block, str(exc))
            assert not compiler.lines, "reject invalid launches before emitting instructions"
        else:
            assert error is None, (grid, block, "should have failed")
            assert "mad.lo.u32" in ptx and "setp.ge.u32" in ptx

    # Keep the existing cooperative schedule restrictions, not just hardware limits.
    for threads, error in ((128, "does not match"), (3, "partial cooperative blocks")):
        cooperative = replace(kernel, body=(replace(kernel.body[0], block_threads=threads),))
        descriptor = launch | {"block": [64 if threads == 128 else 3, 1, 1], "grid": [64, 1, 1]}
        try:
            Compiler(graph, cooperative, descriptor).compile()
        except (ValueError, NotImplementedError) as exc:
            assert error in str(exc)
        else:
            raise AssertionError("invalid cooperative launch was accepted")

    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        target = folder / ir.name
        target.write_text(ir.read_text())
        bad_launch = launch | {"block": [1025, 1, 1]}
        (folder / "meta.json").write_text(json.dumps({"kernels": [bad_launch]}))
        result = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(target)],
                                capture_output=True, text=True)
        assert result.returncode != 0 and result.stdout == "" and "block.x" in result.stderr
    print("PASS: launch dimensions, range boundaries, overflow, coverage, cooperation, clean CLI rejection", flush=True)


def main(remote=False):
    check_vector_loads()
    check_launch_ranges()
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
        assert ptx.count("ld.global.v2.f32") == 2
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
