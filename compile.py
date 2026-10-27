#!/usr/bin/env python3
"""Direct Kernel IR -> PTX compiler for static f32 kernels.

Emmy is used for typed IR loading, never CUDA rendering or PTX generation.
Only the matching launch descriptor in adjacent meta.json is consulted.
"""
import json
import math
from pathlib import Path
import re
import struct
import sys

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import (
    CpAsyncCommit, CpAsyncCopy, CpAsyncWait, KernelOp, Smem, Sync, Tile, TreeHalve, WarpShuffle,
)
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Init, Let, Load, Loop, StridedLoop, Write


class Compiler:
    def __init__(self, graph, kernel, launch):
        self.kernel = kernel
        self.launch = launch
        self.buffers = {t.name: t for n in graph.nodes.values() for t in n.outputs}
        self.counts = dict(r=0, rd=0, f=0, p=0)
        self.lines = []
        self.values = {}
        self.indices = {}
        self.pointers = {}
        self.shapes = {name: tuple(d.as_static() for d in t.shape) for name, t in self.buffers.items()}
        self.shared = {}
        self.shared_bytes = 0
        self.dynamic_shared = launch["smem_bytes"] > 0
        self.declarations = []
        self.labels = 0

    def emit(self, line):
        self.lines.append(f"    {line}")

    def instruction(self, op, *args, kind="f"):
        reg = f"%{kind}{self.counts[kind]}"
        self.counts[kind] += 1
        self.emit(f"{op} {', '.join([reg, *map(str, args)])};")
        return reg

    def label(self):
        self.labels += 1
        return f"L{self.labels}"

    @staticmethod
    def literal(value):
        return "0f" + struct.pack(">f", float(value)).hex().upper()

    def predicate(self, expr):
        if isinstance(expr, BinaryExpr):
            if expr.op in ("&&", "||"):
                op = "and.pred" if expr.op == "&&" else "or.pred"
                return self.instruction(op, self.predicate(expr.left), self.predicate(expr.right), kind="p")
            comparisons = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
            if expr.op in comparisons:
                return self.instruction(f"setp.{comparisons[expr.op]}.s32",
                                        self.index(expr.left), self.index(expr.right), kind="p")
        raise NotImplementedError(f"predicate: {expr!r}")

    def index(self, expr):
        if isinstance(expr, Var):
            return self.indices[expr.name]
        if isinstance(expr, Literal) and expr.dtype == "int":
            if not isinstance(expr.value, int) or not -(2**31) <= expr.value < 2**31:
                raise NotImplementedError("indices must fit signed 32 bits")
            return str(expr.value)
        if isinstance(expr, BinaryExpr):
            left, right = self.index(expr.left), self.index(expr.right)
            ops = {"+": "add.s32", "-": "sub.s32", "*": "mul.lo.s32"}
            if expr.op in ops:
                return self.instruction(ops[expr.op], left, right, kind="r")
            if expr.op in ("/", "//", "%"):
                # IR integer division floors; PTX signed division truncates.
                q = self.instruction("div.s32", left, right, kind="r")
                r = self.instruction("rem.s32", left, right, kind="r")
                signs = self.instruction("xor.b32", left, right, kind="r")
                opposite = self.instruction("setp.lt.s32", signs, 0, kind="p")
                nonzero = self.instruction("setp.ne.s32", r, 0, kind="p")
                adjust = self.instruction("and.pred", opposite, nonzero, kind="p")
                lower = self.instruction("sub.s32", q, 1, kind="r")
                q = self.instruction("selp.b32", lower, q, adjust, kind="r")
                if expr.op != "%":
                    return q
                product = self.instruction("mul.lo.s32", q, right, kind="r")
                return self.instruction("sub.s32", left, product, kind="r")
        if isinstance(expr, TernaryExpr):
            return self.instruction("selp.b32", self.index(expr.if_true), self.index(expr.if_false),
                                    self.predicate(expr.cond), kind="r")
        raise NotImplementedError(f"index expression: {expr!r}")

    def space(self, name):
        return "shared" if name in self.shared else "global"

    def address(self, name, index):
        shape = self.shapes[name]
        # Empty indices mean offset zero; a single index is already flat.
        flat = "0"
        if len(index) == 1:
            flat = self.index(index[0])
        elif index:
            if len(index) != len(shape):
                raise NotImplementedError(f"index rank does not match buffer {name}")
            for extent, coordinate in zip(shape, index, strict=True):
                flat = self.instruction("mul.lo.s32", flat, extent, kind="r")
                flat = self.instruction("add.s32", flat, self.index(coordinate), kind="r")
        if name in self.shared:
            offset = self.instruction("mul.lo.s32", flat, 4, kind="r")
            return self.instruction("add.u32", self.pointers[name], offset, kind="r")
        offset = self.instruction("mul.wide.s32", flat, 4, kind="rd")
        return self.instruction("add.u64", self.pointers[name], offset, kind="rd")

    @staticmethod
    def check_dtype(dtype):
        if dtype is not None and dtype.name != "f32":
            raise NotImplementedError(f"only f32 values are supported, got {dtype.name}")

    def operation(self, name, args):
        ops = {
            "add": ("add.rn.f32", 2), "subtract": ("sub.rn.f32", 2),
            "multiply": ("mul.rn.f32", 2), "divide": ("div.rn.f32", 2),
            "maximum": ("max.NaN.f32", 2), "minimum": ("min.NaN.f32", 2),
            "sqrt": ("sqrt.rn.f32", 1), "relu": ("max.NaN.f32", 1),
            "reciprocal": ("div.rn.f32", 1), "copy": ("mov.f32", 1),
            "exp": ("ex2.approx.f32", 1),
        }
        if name not in ops:
            raise NotImplementedError(f"elementwise operation: {name}")
        opcode, arity = ops[name]
        if len(args) != arity:
            raise ValueError(f"wrong arity for {name}")
        if name == "relu":
            args = [*args, self.literal(0)]
        elif name == "reciprocal":
            args = [self.literal(1), *args]
        elif name == "exp":
            args = [self.instruction("mul.rn.f32", args[0], self.literal(math.log2(math.e)))]
        # Explicit rounding preserves separate multiply/add semantics.
        return self.instruction(opcode, *args)

    def body(self, body):
        for stmt in body:
            self.statement(stmt)

    def loop(self, stmt):
        outer_values, outer_indices = self.values.copy(), self.indices.copy()
        accumulators = {a.name: a for a in stmt.body if isinstance(a, Accum)}
        for name, accum in accumulators.items():
            self.check_dtype(accum.dtype)
            if stmt.seed:
                if accum.op.identity is None:
                    raise NotImplementedError(f"no identity for {accum.op.name}")
                self.values[name] = self.instruction("mov.f32", self.literal(accum.op.identity))
            elif name not in self.values:
                raise ValueError(f"unseeded accumulator: {name}")
        start, step, end = Literal(0, "int"), Literal(1, "int"), stmt.axis.extent.expr
        if isinstance(stmt, StridedLoop):
            start, step = stmt.start, stmt.step
            if isinstance(step, int):
                step = Literal(step, "int")
            end = stmt.end if stmt.end is not None else end
        if isinstance(step, Literal) and step.value <= 0:
            raise NotImplementedError("only increasing loops are supported")
        counter = self.instruction("mov.s32", self.index(start), kind="r")
        self.indices[stmt.axis.name] = counter
        head, done = self.label(), self.label()
        self.emit(f"{head}:")
        stop = self.instruction("setp.ge.s32", counter, self.index(end), kind="p")
        self.emit(f"@{stop} bra {done};")
        self.body(stmt.body)
        self.emit(f"add.s32 {counter}, {counter}, {self.index(step)};")
        self.emit(f"bra {head};")
        self.emit(f"{done}:")
        carried = {name: self.values[name] for name in accumulators}
        self.values = outer_values | carried
        self.indices = outer_indices

    def combine(self, stmt):
        outer = self.values.copy()
        self.body(stmt.combine_states)
        for name in stmt.state:
            self.emit(f"mov.f32 {outer[name]}, {self.values[name]};")
        self.values = outer

    def tree(self, stmt):
        self.check_dtype(stmt.dtype)
        if stmt.length <= 0 or stmt.length & (stmt.length - 1):
            raise ValueError("TreeHalve length must be a power of two")
        outer = self.values.copy()
        tid = Var(stmt.tid_var)
        scale = stmt.inner[1] if stmt.inner else 1
        inner = Var(stmt.inner[0]) if stmt.inner else Literal(0, "int")
        base = tid * scale + inner
        half = stmt.length // 2
        while half:
            done = self.label()
            skip = self.instruction("setp.ge.s32", self.index(tid), half, kind="p")
            self.emit(f"@{skip} bra {done};")
            # One writer per warp slot avoids redundant same-address writes.
            if stmt.tid_var == "warp" and stmt.inner is None:
                nonleader = self.instruction("setp.ne.u32", self.indices["lane"], 0, kind="p")
                self.emit(f"@{nonleader} bra {done};")
            addresses = []
            for buf, left, right in zip(stmt.bufs, stmt.state, stmt.state_b, strict=True):
                address = self.address(buf, (base,))
                other = self.address(buf, (base + half * scale,))
                addresses.append(address)
                self.values[left] = self.instruction("ld.shared.f32", f"[{address}]")
                self.values[right] = self.instruction("ld.shared.f32", f"[{other}]")
            self.combine(stmt)
            for address, name in zip(addresses, stmt.state, strict=True):
                self.emit(f"st.shared.f32 [{address}], {self.values[name]};")
            self.emit(f"{done}:")
            self.statement(Sync(barrier_id=stmt.barrier_id, count=stmt.barrier_count))
            self.values = outer.copy()
            half //= 2
        for buf, name in zip(stmt.bufs, stmt.state, strict=True):
            address = self.address(buf, (inner,))
            self.emit(f"ld.shared.f32 {outer[name]}, [{address}];")
        self.values = outer

    def statement(self, stmt):
        if isinstance(stmt, Load):
            self.check_dtype(stmt.dtype)
            address = self.address(stmt.input, stmt.index)
            # ponytail: scalarize grouped accesses; vectorize when speed matters.
            for i, name in enumerate(stmt.names):
                self.values[name] = self.instruction(f"ld.{self.space(stmt.input)}.f32", f"[{address}+{4 * i}]")
        elif isinstance(stmt, Assign):
            self.check_dtype(stmt.dtype)
            self.values[stmt.name] = self.operation(stmt.op.name, [self.values[name] for name in stmt.args])
        elif isinstance(stmt, Accum):
            self.check_dtype(stmt.dtype)
            value = self.operation(stmt.op.name, [self.values[stmt.base or stmt.name], self.values[stmt.value]])
            self.emit(f"mov.f32 {self.values[stmt.name]}, {value};")
        elif isinstance(stmt, Init):
            self.check_dtype(stmt.dtype)
            self.values[stmt.name] = self.instruction("mov.f32", self.literal(stmt.identity))
        elif isinstance(stmt, Let):
            self.check_dtype(stmt.dtype)
            if not isinstance(stmt.value, Literal):
                raise NotImplementedError("only literal Let bindings are supported")
            self.values[stmt.name] = self.instruction("mov.f32", self.literal(stmt.value.value))
        elif isinstance(stmt, Write):
            self.check_dtype(stmt.value_dtype)
            if stmt.swizzle != "NONE":
                raise NotImplementedError("swizzled writes are not supported yet")
            address = self.address(stmt.output, stmt.index)
            if stmt.atomic:
                if self.space(stmt.output) != "global" or len(stmt.values) != 1:
                    raise NotImplementedError("only scalar global atomic additions are supported")
                # The host applies meta.json's zero_outputs before each launch.
                self.emit(f"red.global.add.f32 [{address}], {self.values[stmt.values[0]]};")
            else:
                for i, name in enumerate(stmt.values):
                    self.emit(f"st.{self.space(stmt.output)}.f32 [{address}+{4 * i}], {self.values[name]};")
        elif isinstance(stmt, CpAsyncCopy):
            if stmt.swizzle != "NONE" or stmt.nbytes not in (4, 8, 16):
                raise NotImplementedError("unsupported async copy layout/size")
            if self.space(stmt.smem) != "shared" or self.space(stmt.src) != "global":
                raise ValueError("cp.async must copy global to shared memory")
            index = stmt.smem_index
            if stmt.lane_index is not None:
                index = tuple(a + b for a, b in zip(index, stmt.lane_index, strict=True))
            dst, src = self.address(stmt.smem, index), self.address(stmt.src, stmt.src_index)
            cache = "cg" if stmt.nbytes == 16 else "ca"
            self.emit(f"cp.async.{cache}.shared.global [{dst}], [{src}], {stmt.nbytes};")
        elif isinstance(stmt, CpAsyncCommit):
            self.emit("cp.async.commit_group;")
        elif isinstance(stmt, CpAsyncWait):
            if not 0 <= stmt.group <= 7:
                raise ValueError("cp.async wait group must be in 0..7")
            self.emit(f"cp.async.wait_group {stmt.group};")
        elif isinstance(stmt, (Loop, StridedLoop)):
            self.loop(stmt)
        elif isinstance(stmt, Cond):
            otherwise, done = self.label(), self.label()
            cond = self.predicate(stmt.cond)
            self.emit(f"@!{cond} bra {otherwise};")
            outer = self.values.copy()
            self.body(stmt.body)
            self.emit(f"bra {done};")
            self.emit(f"{otherwise}:")
            self.values = outer.copy()
            self.body(stmt.else_body)
            self.emit(f"{done}:")
            self.values = outer
        elif isinstance(stmt, Smem):
            if stmt.dtype != "float" or stmt.name in self.shared:
                raise NotImplementedError("expected distinct f32 shared buffers")
            symbol = f"shared_{len(self.shared)}"
            self.shared[stmt.name] = symbol
            self.shapes[stmt.name] = tuple(stmt.extents)
            size, align = 4 * math.prod(stmt.extents), max(4, stmt.align)
            if size <= 0 or align & (align - 1) or align > 16:
                raise NotImplementedError("shared size/alignment is unsupported")
            offset = (self.shared_bytes + align - 1) // align * align
            self.shared_bytes = offset + size
            if self.dynamic_shared:
                base = self.instruction("mov.u32", "shared_pool", kind="r")
                self.pointers[stmt.name] = self.instruction("add.u32", base, offset, kind="r")
            else:
                self.declarations.append(f"    .shared .align {align} .b8 {symbol}[{size}];")
                self.pointers[stmt.name] = self.instruction("mov.u32", symbol, kind="r")
        elif isinstance(stmt, Sync):
            if stmt.warp:
                mask = self.instruction("activemask.b32", kind="r")
                self.emit(f"bar.warp.sync {mask};")
            else:
                if stmt.barrier_id and stmt.count is None:
                    raise ValueError("named barriers need a thread count")
                count = f", {stmt.count}" if stmt.count is not None else ""
                self.emit(f"bar.sync {stmt.barrier_id}{count};")
        elif isinstance(stmt, WarpShuffle):
            self.check_dtype(stmt.dtype)
            if not 1 <= stmt.length <= 32 or stmt.length & (stmt.length - 1):
                raise ValueError("WarpShuffle length must be a power of two up to 32")
            mask = self.instruction("activemask.b32", kind="r")
            half = stmt.length // 2
            while half:
                for left, right in zip(stmt.state, stmt.state_b, strict=True):
                    bits = self.instruction("mov.b32", self.values[left], kind="r")
                    bits = self.instruction("shfl.sync.bfly.b32", bits, half, "0x1f", mask, kind="r")
                    self.values[right] = self.instruction("mov.b32", bits)
                self.combine(stmt)
                half //= 2
        elif isinstance(stmt, TreeHalve):
            self.tree(stmt)
        else:
            raise NotImplementedError(f"statement: {type(stmt).__name__}")

    def compile(self):
        kernel, launch = self.kernel, self.launch
        if not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", kernel.name):
            raise ValueError("kernel entry must be an ASCII PTX identifier")
        if kernel.serial or len(kernel.body) != 1 or not isinstance(kernel.body[0], Tile):
            raise NotImplementedError("expected one non-serial Tile")
        tile = kernel.body[0]
        if tile.aux_threads:
            raise NotImplementedError("auxiliary threads are not supported yet")
        # ponytail: omit CTA raster reordering (cache locality only); tune later.
        if tile.block_threads is not None and tile.block_threads != launch["block"][0]:
            raise ValueError("launch block size does not match cooperative Tile")
        if tile.block_threads and tile.n_elements % tile.block_threads:
            raise NotImplementedError("partial cooperative blocks could skip a barrier")
        if launch["grid"][1:] != [1, 1] or launch["block"][1:] != [1, 1]:
            raise NotImplementedError("only 1-D launches are supported")
        if not 0 < tile.n_elements < 2**31 or any(e <= 0 for e in tile.extents):
            raise NotImplementedError("expected positive, signed-32-bit tile size")
        if launch["grid"][0] * launch["block"][0] < tile.n_elements:
            raise ValueError("launch does not cover the Tile")

        order = launch["arg_order"]
        if len(order) != len(set(order)):
            raise ValueError("duplicate kernel argument")
        for i, name in enumerate(order):
            tensor = self.buffers[name]
            self.check_dtype(tensor.dtype)
            size = math.prod(d.as_static() for d in tensor.shape)
            if not 0 < size < 2**31:
                raise NotImplementedError("buffer size must fit signed 32 bits")
            ptr = self.instruction("ld.param.u64", f"[param_{i}]", kind="rd")
            self.pointers[name] = self.instruction("cvta.to.global.u64", ptr, kind="rd")

        block = self.instruction("mov.u32", "%ctaid.x", kind="r")
        width = self.instruction("mov.u32", "%ntid.x", kind="r")
        thread = self.instruction("mov.u32", "%tid.x", kind="r")
        self.indices["lane"] = self.instruction("and.b32", thread, 31, kind="r")
        self.indices["warp"] = self.instruction("shr.u32", thread, 5, kind="r")
        linear = self.instruction("mad.lo.u32", block, width, thread, kind="r")
        outside = self.instruction("setp.ge.u32", linear, tile.n_elements, kind="p")
        self.emit(f"@{outside} bra DONE;")
        for axis, extent in reversed(list(zip(tile.axes, tile.extents, strict=True))):
            self.indices[axis.name] = self.instruction("rem.u32", linear, extent, kind="r")
            linear = self.instruction("div.u32", linear, extent, kind="r")
        self.body(tile.body)
        shared_limit = launch["smem_bytes"] if self.dynamic_shared else 48 * 1024
        if self.shared_bytes > shared_limit:
            raise ValueError("shared buffers exceed the prescribed allocation")
        self.emit("DONE:")
        self.emit("ret;")

        params = ",\n".join(f"    .param .u64 param_{i}" for i in range(len(order)))
        types = dict(r="b32", rd="b64", f="f32", p="pred")
        registers = [f"    .reg .{types[k]} %{k}<{n}>;" for k, n in self.counts.items() if n]
        pool = [".extern .shared .align 16 .b8 shared_pool[];"] if self.dynamic_shared else []
        return "\n".join([
            ".version 8.0", ".target sm_89", ".address_size 64", *pool, "",
            f".visible .entry {kernel.name}(\n{params}\n)", "{",
            *self.declarations, *registers, *self.lines, "}", "",
        ])


def compile_file(path):
    path = Path(path)
    graph = Graph.from_dict(json.loads(path.read_text()))
    kernels = [n.op for n in graph.nodes.values() if isinstance(n.op, KernelOp)]
    if len(kernels) != 1:
        raise ValueError("expected exactly one KernelOp in the IR")
    kernel = kernels[0]
    meta = json.loads(path.with_name("meta.json").read_text())
    launches = [k for k in meta["kernels"] if k["entry"] == kernel.name]
    if len(launches) != 1:
        raise ValueError("expected one matching entry in meta.json")
    return Compiler(graph, kernel, launches[0]).compile()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python compile.py path/to/kernel.ir.json")
    try:
        ptx = compile_file(sys.argv[1])
    except (ValueError, TypeError, KeyError, OSError, NotImplementedError) as exc:
        sys.exit(f"compile.py: {exc}")
    print(ptx, end="")
