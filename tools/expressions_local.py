"""Extra development cases; generate with tools/generate.py, not compile.py.

These are NOT a hidden-set proxy or an untouched holdout once used for debugging.
NumPy references are independent of our PTX lowering. Torch is generation-only.
"""
import numpy as np

from expressions_open import REDUCE_KNOBS, _softmax

OPEN = [
    dict(id="v01_single", code="torch.randn(1) + torch.randn(1)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="v02_odd_product", code="torch.randn(127) * torch.randn(127) - 2.25",
         ref=lambda a, b: [a * b - np.float32(2.25)], rtol=0, atol=0),
    dict(id="v03_block_tail", code="torch.randn(257) + torch.randn(257)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="v04_relu_tail", code="torch.relu(torch.randn(1031) * 1.25 - 0.375)",
         ref=lambda x: [np.maximum(x * np.float32(1.25) - np.float32(0.375), 0)],
         rtol=0, atol=0),
    dict(id="v05_broadcast", code="torch.randn(13, 37) + torch.randn(37)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="v06_sqrt_div", code="x = torch.randn(513); x / torch.sqrt(x * x + 0.5)",
         ref=lambda x: [x / np.sqrt(x * x + np.float32(0.5))], rtol=1e-6, atol=1e-6),
    dict(id="v07_large_sub", code="torch.randn(8193) - torch.randn(8193)",
         ref=lambda a, b: [a - b], rtol=0, atol=0),
    dict(id="v08_vector_constants", code="torch.randn(4096) * -1.75 + 0.125",
         ref=lambda x: [x * np.float32(-1.75) + np.float32(0.125)], rtol=0, atol=0),
    dict(id="v09_partial_warp", code="torch.randn(7).sum()",
         ref=lambda x: [x.astype(np.float64).sum().astype(np.float32)],
         rtol=1e-4, atol=1e-4, knobs="WORK=t8,REDUCE=coop"),
    dict(id="v10_sum_tail", code="torch.randn(257).sum()",
         ref=lambda x: [x.astype(np.float64).sum().astype(np.float32)],
         rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="v11_big_sum_tail", code="torch.randn(65537).sum()",
         ref=lambda x: [x.astype(np.float64).sum().astype(np.float32)],
         rtol=1e-3, atol=1e-3, knobs="WORK=t256,REDUCE=coop"),
    dict(id="v12_row_sum", code="torch.randn(13, 259).sum(-1)",
         ref=lambda x: [x.astype(np.float64).sum(-1).astype(np.float32)],
         rtol=1e-4, atol=1e-4, knobs="WORK=t64,REDUCE=coop"),
    dict(id="v13_column_sum", code="torch.randn(17, 35).sum(0)",
         ref=lambda x: [x.astype(np.float64).sum(0).astype(np.float32)],
         rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="v14_row_mean", code="torch.randn(5, 513).mean(-1)",
         ref=lambda x: [x.astype(np.float64).mean(-1).astype(np.float32)],
         rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="v15_row_max", code="torch.randn(6, 97).amax(-1)",
         ref=lambda x: [x.max(-1)], rtol=0, atol=0, knobs="WORK=t32,REDUCE=coop"),
    # Emmy 0.3.11 cannot trace prod/amin here; these use supported expressions.
    dict(id="v16_row_squares", code="x = torch.randn(7, 47); (x * x).sum(-1)",
         ref=lambda x: [(x * x).astype(np.float64).sum(-1).astype(np.float32)],
         rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="v17_softmax_tail", code="torch.softmax(torch.randn(7, 259), dim=-1)",
         ref=lambda x: [_softmax(x.astype(np.float64)).astype(np.float32)],
         rtol=1e-3, atol=1e-5, knobs=REDUCE_KNOBS),
    dict(id="v18_rms_tail", code="x = torch.randn(3, 513); w = torch.randn(513); "
         "x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5) * w",
         ref=lambda x, w: [(x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-5) * w).astype(np.float32)],
         rtol=1e-3, atol=1e-4, knobs="WORK=t64,REDUCE=coop"),
    dict(id="v19_l2_small_warp", code="x = torch.randn(5, 97); x / torch.sqrt((x * x).sum(-1, keepdim=True))",
         ref=lambda x: [(x / np.sqrt((x.astype(np.float64) ** 2).sum(-1, keepdims=True))).astype(np.float32)],
         rtol=1e-3, atol=1e-4, knobs="WORK=t8,REDUCE=coop"),
    dict(id="v20_max", code="torch.randn(1043).amax()",
         ref=lambda x: [x.max()], rtol=0, atol=0, knobs="WORK=t256,REDUCE=coop"),
    dict(id="v21_matmul_rect", code="torch.randn(24, 40) @ torch.randn(40, 20)",
         ref=lambda a, b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)],
         rtol=1e-3, atol=1e-3),
    dict(id="v22_matmul_tail", code="torch.randn(13, 17) @ torch.randn(17, 19)",
         ref=lambda a, b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)],
         rtol=1e-3, atol=1e-3),
    dict(id="v23_attention_rect", code="q = torch.randn(12, 8); kt = torch.randn(8, 20); "
         "v = torch.randn(20, 6); torch.softmax(q @ kt / 2.828427, dim=-1) @ v",
         ref=lambda q, kt, v: [(_softmax((q.astype(np.float64) @ kt.astype(np.float64)) / 2.828427)
                               @ v.astype(np.float64)).astype(np.float32)], rtol=1e-3, atol=1e-4),
    dict(id="v24_matmul_tiles", code="torch.randn(64, 96) @ torch.randn(96, 48)",
         ref=lambda a, b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)],
         rtol=1e-3, atol=1e-3),
]
