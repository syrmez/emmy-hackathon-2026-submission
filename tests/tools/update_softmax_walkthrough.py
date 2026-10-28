"""Refresh embedded source/PTX evidence: python tests/tools/update_softmax_walkthrough.py [--check]."""
import ast
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from compile import compile_file

source = (ROOT / 'compile.py').read_text()
ptx = compile_file(ROOT / 'tests/problems/f02_row_softmax/k_softmax_05263d.ir.json')
methods = {n.name: n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)}


def method(name):
    n = methods[name]
    return '\n'.join(f'{i + 1:3} {line}' for i, line in enumerate(source.splitlines())
                     if n.lineno - 1 <= i < n.end_lineno)


def between(text, start, end):
    return text[text.index(start):text.index(end, text.index(start))].rstrip()


load_lines = ptx.splitlines()
load = next(i for i, line in enumerate(load_lines) if 'ld.global.f32' in line)
evidence = {
    'sha256': hashlib.sha256(source.encode()).hexdigest(),
    'load': {
        'source': method('address') + '\n\n' + between(source, '    def statement(self, stmt):', '        elif isinstance(stmt, Assign):'),
        'ptx': '\n'.join(load_lines[load - 8:load + 1]),
    },
    'shuffle': {
        'source': between(source, '        elif isinstance(stmt, WarpShuffle):', '        elif isinstance(stmt, TreeHalve):') + '\n\n' + method('combine'),
        'ptx': between(ptx, '    activemask.b32', load_lines[next(i for i, line in enumerate(load_lines) if 'shfl.sync.bfly.b32' in line and ', 8,' in line) - 1]),
    },
    'tree': {
        'source': method('tree'),
        'ptx': between(ptx, '    bar.sync 0;', '    div.rn.f32'),
    },
}
# Excerpts are contiguous actual output, not illustrative hand-written PTX.
start, end = '<pre id="compiler-evidence" hidden>', '</pre><!-- compiler-evidence -->'
page = ROOT / 'softmax.html'
text = page.read_text()
a, b = text.index(start) + len(start), text.index(end)
updated = text[:a] + html.escape(json.dumps(evidence)) + text[b:]
if sys.argv[1:] == ['--check']:
    assert updated == text, 'Run python tests/tools/update_softmax_walkthrough.py to refresh source/PTX evidence'
    print('PASS: embedded compiler excerpts and PTX match current source and actual compilation')
elif not sys.argv[1:]:
    page.write_text(updated)
else:
    sys.exit('usage: python tests/tools/update_softmax_walkthrough.py [--check]')
