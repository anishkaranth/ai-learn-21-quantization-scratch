"""Matplotlib SVG plots + RESULTS.md writer for the quantization smoke run."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from svg_utils import minify_svg  # noqa: E402

plt.rcParams.update({"svg.hashsalt": "ai-learn-21", "svg.fonttype": "none", "font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"], "axes.unicode_minus": False})
COL = {"sym/per_tensor": "#126782", "sym/per_channel": "#2a9d8f", "asym/per_tensor": "#e76f51", "asym/per_channel": "#e9c46a"}


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None})
    plt.close(fig)
    path.write_text(minify_svg(buf.getvalue()), encoding="utf-8")
    return path.name


def make_plots(out: Path, m: Dict[str, Any], err: Dict[str, np.ndarray]) -> List[str]:
    names, bits = [], m["config"]["bits"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for s, c in COL.items():
        ax.plot(bits, [r["acc"] for r in m["sweep"] if r["scheme"] == s], color=c, lw=1.8, label=s)
    ax.axhline(m["fp32"]["acc"], color="grey", ls="--", lw=1, label=f"fp32 ({m['fp32']['acc']:.3f})")
    ax.set_xlabel("weight bits")
    ax.set_ylabel("test accuracy")
    ax.set_xticks(bits)
    ax.invert_xaxis()
    ax.set_title("Post-training quantization: accuracy vs bits")
    ax.legend(fontsize=8)
    names.append(_save(fig, out / "accuracy_vs_bits.svg"))

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for s, c in COL.items():
        rs = [r for r in m["sweep"] if r["scheme"] == s]
        ax.plot([r["size_bytes"] / 1024 for r in rs], [r["acc"] for r in rs], color=c, lw=1.2, label=s)
    ax.axvline(m["fp32"]["size_bytes"] / 1024, color="grey", ls="--", lw=1, label="fp32 size")
    ax.set_xlabel("model size (KiB, weights + scales/zero-points + fp32 biases)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy vs model size (points = 8, 6, 4, 3, 2 bits)")
    ax.legend(fontsize=8)
    names.append(_save(fig, out / "accuracy_vs_size.svg"))

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, b in zip(axes, (8, 4)):
        for g, c in (("per_tensor", "#126782"), ("per_channel", "#e76f51")):
            e = err[f"int{b} {g}"]
            ax.hist(e, bins=40, histtype="step", color=c, lw=1.4, label=f"{g} (std {e.std():.1e})")
        ax.set_title(f"int{b} symmetric: W1 error (w_hat - w)")
        ax.set_xlabel("quantization error")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("weights")
    names.append(_save(fig, out / "error_histograms.svg"))
    return names


def write_results_md(path: Path, m: Dict[str, Any], plots: List[str]) -> None:
    c, h, fp = m["config"], m["headline"], m["fp32"]
    L = [f"# Results -- {m['project']}", "",
         f"**Seed:** `{m['seed']}` | MLP {'->'.join(map(str, c['sizes']))} ({m['n_params']} params) | train n={c['n_train']} | "
         f"test n={c['n_test']} | fp32 test acc **{fp['acc']:.4f}**, size {fp['size_bytes']} B", "",
         "Weight-only post-training quantization (biases and activations stay fp32).", "",
         "## Accuracy by bits and scheme", "", "| bits | " + " | ".join(COL) + " |", "|---:|" + "---:|" * len(COL)]
    for b in c["bits"]:
        L.append(f"| {b} | " + " | ".join(f"{r['acc']:.4f}" for s in COL for r in m["sweep"] if r["bits"] == b and r["scheme"] == s) + " |")
    L += ["", "## Size and weight MSE (symmetric per-tensor vs asymmetric per-channel)", "",
          "| bits | size sym/per_tensor (B) | ratio vs fp32 | MSE sym/per_tensor | size asym/per_channel (B) | MSE asym/per_channel |", "|---:|---:|---:|---:|---:|---:|"]
    for b in c["bits"]:
        a = next(r for r in m["sweep"] if r["bits"] == b and r["scheme"] == "sym/per_tensor")
        d = next(r for r in m["sweep"] if r["bits"] == b and r["scheme"] == "asym/per_channel")
        L.append(f"| {b} | {a['size_bytes']} | {a['size_bytes'] / fp['size_bytes']:.3f} | {a['weight_mse']:.2e} | {d['size_bytes']} | {d['weight_mse']:.2e} |")
    L += ["", f"## Outlier channel test (hidden unit {c['outlier_channel']} scaled x{c['outlier_factor']:g}, next layer /{c['outlier_factor']:g})", "",
          f"The rescale is function-preserving (fp32 acc after rescale: {m['outlier']['fp32_acc_after_rescale']:.4f}), but one big column stretches the per-tensor range.", "",
          "| bits | " + " | ".join(COL) + " |", "|---:|" + "---:|" * len(COL)]
    for b in (8, 4):
        L.append(f"| {b} | " + " | ".join(f"{r['acc']:.4f}" for s in COL for r in m["outlier"]["rows"] if r["bits"] == b and r["scheme"] == s) + " |")
    L += ["", "## Takeaways", "",
          f"- int8 is effectively lossless here ({h['int8_sym_per_tensor_acc']:.4f} vs fp32 {h['fp32_acc']:.4f}) at {h['size_ratio_int8']:.1%} of the fp32 size.",
          f"- int4 costs little: {h['int4_sym_per_tensor_acc']:.4f} per-tensor, {h['int4_sym_per_channel_acc']:.4f} per-channel, at {h['size_ratio_int4']:.1%} of the size.",
          f"- At 2 bits symmetric per-tensor collapses to {h['int2_sym_per_tensor_acc']:.4f}; asymmetric per-channel still reaches {h['int2_asym_per_channel_acc']:.4f}.",
          f"- With an outlier channel, int4 per-tensor drops to {h['outlier_int4_per_tensor_acc']:.4f} while per-channel keeps {h['outlier_int4_per_channel_acc']:.4f}. "
          "This is why real LLM quantizers use per-channel or per-group scales.", "",
          "## Plots", ""] + [f"![{p}]({p})" for p in plots] + ["", f"Wall time: {m['wall_time_s']:.2f}s on CPU."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
