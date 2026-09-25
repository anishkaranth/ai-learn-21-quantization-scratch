"""Tiny NumPy MLP (ReLU hidden layers, softmax output) with manual backprop and Adam."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

Params = Dict[str, np.ndarray]


def make_data(n: int, seed: int, d: int = 20, n_classes: int = 8, clusters_per_class: int = 3):
    """Gaussian clusters in d dims; several clusters per class so the boundary is non-linear."""
    rng = np.random.default_rng(1000)  # fixed centres shared by train/test
    centres = rng.normal(0, 1.2, (n_classes * clusters_per_class, d))
    r = np.random.default_rng(seed)
    k = r.integers(0, len(centres), n)
    X = centres[k] + r.normal(0, 1.0, (n, d))
    return X.astype(np.float64), (k % n_classes).astype(np.int64)


def init_mlp(sizes: List[int], rng: np.random.Generator) -> Params:
    p = {}
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        p[f"W{i}"] = rng.normal(0, np.sqrt(2.0 / a), (a, b))
        p[f"b{i}"] = np.zeros(b)
    return p


def n_layers(p: Params) -> int:
    return sum(1 for k in p if k.startswith("W"))


def forward(p: Params, X: np.ndarray):
    """Returns (logits, cache of layer outputs)."""
    h, cache = X, [X]
    L = n_layers(p)
    for i in range(L):
        z = h @ p[f"W{i}"] + p[f"b{i}"]
        h = np.maximum(z, 0) if i < L - 1 else z
        cache.append(h)
    return h, cache


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def backward(p: Params, cache, dlogits: np.ndarray) -> Params:
    g, d = {}, dlogits
    for i in reversed(range(n_layers(p))):
        g[f"W{i}"] = cache[i].T @ d
        g[f"b{i}"] = d.sum(0)
        if i:
            d = (d @ p[f"W{i}"].T) * (cache[i] > 0)
    return g


class Adam:
    def __init__(self, p: Params, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999):
        self.lr, self.b1, self.b2, self.t = lr, b1, b2, 0
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}

    def step(self, p: Params, g: Params) -> None:
        self.t += 1
        for k in p:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            p[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def train(p: Params, X, y, epochs: int, lr: float, batch: int, rng) -> List[float]:
    opt, n, hist = Adam(p, lr), len(X), []
    for _ in range(epochs):
        idx, tot = rng.permutation(n), 0.0
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            logits, cache = forward(p, X[b])
            P = softmax(logits)
            tot += -np.log(P[np.arange(len(b)), y[b]] + 1e-12).sum()
            P[np.arange(len(b)), y[b]] -= 1
            opt.step(p, backward(p, cache, P / len(b)))
        hist.append(tot / n)
    return hist


def accuracy(p: Params, X, y) -> float:
    return float((forward(p, X)[0].argmax(1) == y).mean())
