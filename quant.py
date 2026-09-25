"""Post-training weight quantization from scratch.

For a float tensor ``w`` and ``b`` bits:

* **symmetric**:  scale = max|w| / (2^(b-1) - 1), zero_point = 0, q = clip(round(w/scale), -(2^(b-1)-1), 2^(b-1)-1)
* **asymmetric**: scale = (max w - min w) / (2^b - 1), zero_point = round(-min w / scale),
  q = clip(round(w/scale) + zero_point, 0, 2^b - 1)

Dequantize with ``w_hat = scale * (q - zero_point)``. **Per-tensor** uses one (scale, zp)
for the whole matrix; **per-channel** uses one per output column (axis 0 is reduced).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def quantize(w: np.ndarray, bits: int, symmetric: bool, per_channel: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = 0 if per_channel else None
    kw = {"axis": axis, "keepdims": True} if per_channel else {}
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        scale = np.abs(w).max(**kw) / qmax
        scale = np.where(scale == 0, 1.0, scale)
        zp = np.zeros_like(scale)
        q = np.clip(np.round(w / scale), -qmax, qmax)
    else:
        qmax = 2 ** bits - 1
        lo, hi = np.minimum(w.min(**kw), 0), np.maximum(w.max(**kw), 0)  # range must contain 0
        scale = (hi - lo) / qmax
        scale = np.where(scale == 0, 1.0, scale)
        zp = np.round(-lo / scale)
        q = np.clip(np.round(w / scale) + zp, 0, qmax)
    return q.astype(np.int32), np.asarray(scale, dtype=np.float64), np.asarray(zp, dtype=np.float64)


def dequantize(q: np.ndarray, scale: np.ndarray, zp: np.ndarray) -> np.ndarray:
    return scale * (q - zp)


def quantize_model(p: Dict[str, np.ndarray], bits: int, symmetric: bool, per_channel: bool):
    """Quantize every weight matrix (biases stay fp32). Returns (dequantized params, size bytes, per-layer errors)."""
    out, size, errs = {}, 0.0, {}
    for k, v in p.items():
        if k.startswith("W"):
            q, s, z = quantize(v, bits, symmetric, per_channel)
            out[k] = dequantize(q, s, z)
            errs[k] = out[k] - v
            n_scales = s.size
            size += v.size * bits / 8 + n_scales * 4 + (0 if symmetric else n_scales * 4)  # fp32 scale (+ zp)
        else:
            out[k] = v.copy()
            size += v.size * 4
    return out, size, errs


def fp32_size(p: Dict[str, np.ndarray]) -> float:
    return float(sum(v.size for v in p.values()) * 4)
