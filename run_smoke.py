#!/usr/bin/env python3
"""Quantization smoke: train fp32 MLP -> PTQ sweep (bits x sym/asym x per-tensor/per-channel) -> outlier test -> results/."""
from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path

import numpy as np

from mlp import accuracy, init_mlp, make_data, train
from quant import fp32_size, quantize_model
from smoke_plots import make_plots, write_results_md

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42
CFG = {"sizes": [20, 128, 64, 8], "n_train": 6000, "n_test": 3000, "epochs": 40, "lr": 2e-3, "batch": 64,
       "bits": [8, 6, 4, 3, 2], "outlier_channel": 0, "outlier_factor": 8.0}
SCHEMES = [("sym", "per_tensor"), ("sym", "per_channel"), ("asym", "per_tensor"), ("asym", "per_channel")]


def _compact(js: str) -> str:
    js = re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", js)
    return re.sub(r"\{\n([^{}\[\]]*?)\n\s*\}", lambda m: "{" + re.sub(r"\s*\n\s*", " ", m.group(1)).strip() + "}", js)


def sweep(p, Xt, yt, bits_list):
    rows = []
    for bits in bits_list:
        for sym, gran in SCHEMES:
            qp, size, errs = quantize_model(p, bits, sym == "sym", gran == "per_channel")
            rows.append({"bits": bits, "scheme": f"{sym}/{gran}", "acc": round(accuracy(qp, Xt, yt), 4),
                         "size_bytes": int(size), "weight_mse": float(f"{np.mean(np.concatenate([e.ravel() for e in errs.values()]) ** 2):.3e}")})
    return rows


def main() -> None:
    t0 = time.perf_counter()
    X, y = make_data(CFG["n_train"], 1)
    Xt, yt = make_data(CFG["n_test"], 2)
    rng = np.random.default_rng(SEED)
    p = init_mlp(CFG["sizes"], rng)
    hist = train(p, X, y, CFG["epochs"], CFG["lr"], CFG["batch"], rng)
    fp_acc, fp_size = accuracy(p, Xt, yt), fp32_size(p)
    rows = sweep(p, Xt, yt, CFG["bits"])

    # Outlier test: scale hidden unit j of layer 0 by c and its outgoing weights by 1/c.
    # ReLU(c*z) = c*ReLU(z) for c > 0, so the fp32 function is identical -- only quantization sees the difference.
    po = copy.deepcopy(p)
    j, c = CFG["outlier_channel"], CFG["outlier_factor"]
    po["W0"][:, j] *= c
    po["b0"][j] *= c
    po["W1"][j, :] /= c
    out_fp = accuracy(po, Xt, yt)
    out_rows = sweep(po, Xt, yt, [8, 4])

    # Error histograms for layer W1 (int8 vs int4, per-tensor vs per-channel, symmetric)
    err = {}
    for bits in (8, 4):
        for gran in ("per_tensor", "per_channel"):
            _, _, e = quantize_model(p, bits, True, gran == "per_channel")
            err[f"int{bits} {gran}"] = e["W1"].ravel()

    by = {(r["bits"], r["scheme"]): r for r in rows}
    m = {"project": "ai-learn-21-quantization-scratch", "seed": SEED, "config": CFG,
         "n_params": int(sum(v.size for v in p.values())), "fp32": {"acc": round(fp_acc, 4), "size_bytes": int(fp_size),
                                                                     "final_train_loss": round(hist[-1], 4)},
         "sweep": rows, "outlier": {"fp32_acc_after_rescale": round(out_fp, 4), "rows": out_rows},
         "error_std_W1": {k: float(f"{v.std():.3e}") for k, v in err.items()},
         "headline": {
             "fp32_acc": round(fp_acc, 4),
             "int8_sym_per_tensor_acc": by[(8, "sym/per_tensor")]["acc"],
             "int4_sym_per_tensor_acc": by[(4, "sym/per_tensor")]["acc"],
             "int4_sym_per_channel_acc": by[(4, "sym/per_channel")]["acc"],
             "int2_sym_per_tensor_acc": by[(2, "sym/per_tensor")]["acc"],
             "int2_asym_per_channel_acc": by[(2, "asym/per_channel")]["acc"],
             "size_ratio_int8": round(by[(8, "sym/per_tensor")]["size_bytes"] / fp_size, 4),
             "size_ratio_int4": round(by[(4, "sym/per_tensor")]["size_bytes"] / fp_size, 4),
             "outlier_int4_per_tensor_acc": next(r["acc"] for r in out_rows if r["bits"] == 4 and r["scheme"] == "sym/per_tensor"),
             "outlier_int4_per_channel_acc": next(r["acc"] for r in out_rows if r["bits"] == 4 and r["scheme"] == "sym/per_channel"),
         }}
    m["wall_time_s"] = round(time.perf_counter() - t0, 2)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(_compact(json.dumps(m, indent=1)) + "\n")
    shot = {"project": m["project"], "seed": SEED, "config": CFG, "fp32": m["fp32"], "headline": m["headline"]}
    (RESULTS / "JSON.shot").write_text(_compact(json.dumps(shot, indent=2)) + "\n")
    plots = make_plots(RESULTS, m, err)
    write_results_md(RESULTS / "RESULTS.md", m, plots)
    json.loads((RESULTS / "JSON.shot").read_text())
    print(json.dumps(m["headline"], indent=2))


if __name__ == "__main__":
    main()
