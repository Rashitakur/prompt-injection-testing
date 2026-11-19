#!/usr/bin/env python
"""Aggregate all runs into CSV tables, LaTeX tables, figures, LaTeX macros and a findings summary.

Outputs:
  results/tables/*.csv
  paper/generated/{tab_data,tab_main,tab_ood,tab_judge,tab_sig}.tex, macros.tex
  paper/figures/{roc_test_id,pareto_latency,fpr_benign,ood_heatmap}.pdf (+ .png)
  results/findings.md     <- read this before editing the paper's claims

    python scripts/aggregate.py --config configs/default.yaml
"""
import argparse
import glob
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from pidetect.metrics import binary_metrics, bootstrap_ci, paired_bootstrap_pvalue
from pidetect.utils import ensure_dir, load_config, load_json, setup_logging

LOG = logging.getLogger("aggregate")

# family -> (order, colour, marker). Colours: validated categorical palette, fixed order.
FAMILIES = {
    "Classical":        (0, "#2a78d6", "s"),
    "Embedding + LR":   (1, "#eb6834", "D"),
    "Fine-tuned":       (2, "#1baf7a", "o"),
    "Perplexity":       (3, "#eda100", "^"),
    "LLM-as-judge":     (4, "#6250d6", "P"),
    "Reference":        (5, "#e87ba4", "X"),
}
TYPE_FAMILY = {"tfidf_lr": "Classical", "embed_lr": "Embedding + LR", "finetune": "Fine-tuned",
               "perplexity": "Perplexity", "llm_judge": "LLM-as-judge", "hf_classifier": "Reference"}
DISPLAY = {
    "tfidf_lr": "TF-IDF + LR", "emb_minilm_lr": "MiniLM-L6 + LR", "emb_bge_lr": "BGE-small + LR",
    "distilbert": "DistilBERT (FT)", "ppl_gpt2-ppl": "GPT-2 PPL", "ppl_gpt2-winppl": "GPT-2 win-PPL",
    "ppl_gpt2-lr": "GPT-2 PPL feats + LR", "judge_qwen1p5b": "Qwen2.5-1.5B judge",
    "judge_qwen7b": "Qwen2.5-7B judge", "judge_llama8b": "Llama-3.1-8B judge",
    "ref_protectai": "ProtectAI DeBERTa-v3$^\\dagger$",
}
SPLIT_DISPLAY = {"test_id": "ID test", "ood_spml": "SPML", "ood_jailbreak": "Jailbreak",
                 "ood_gandalf_dolly": "Gandalf+Dolly", "stress_adv_suffix": "Adv. suffix",
                 "benign_notinject": "NotInject", "benign_noisy": "Noisy benign"}
BOTH_CLASS_METRICS = ["accuracy", "precision", "recall", "f1", "auroc", "auprc", "fpr"]


def sdisp(s: str) -> str:
    return SPLIT_DISPLAY.get(s, s.replace("_", r"\_"))


def disp(v: str) -> str:
    base = v.replace("@0.5", "")
    s = DISPLAY.get(base, base.replace("_", r"\_"))
    return s + (" @0.5" if v.endswith("@0.5") else "")


def key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def fmt(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def fmt_ms(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.2f}" if x < 10 else (f"{x:.1f}" if x < 100 else f"{x:.0f}")


# =============================================================================
def load_runs(res_dir: Path, cfg: dict):
    metas, scores = [], {}
    for mpath in sorted(glob.glob(str(res_dir / "runs" / "*" / "seed*" / "meta.json"))):
        m = load_json(mpath)
        d = Path(mpath).parent
        metas.append(m)
        for sp in glob.glob(str(d / "scores_*.parquet")):
            split = Path(sp).stem.replace("scores_", "")
            scores[(m["variant"], m["seed"], split)] = pd.read_parquet(sp)
    return metas, scores


def compute_long(metas, scores, low_fpr):
    rows = []
    for m in metas:
        thr_list = [(m["variant"], m["threshold"])]
        if m.get("default_threshold") is not None:
            thr_list.append((m["variant"] + "@0.5", m["default_threshold"]))
        for name, thr in thr_list:
            for (v, seed, split), df in scores.items():
                if v != m["variant"] or seed != m["seed"]:
                    continue
                r = binary_metrics(df.label.values, df.score.values, thr, low_fpr)
                r.update({"variant": name, "base_variant": v, "detector": m["detector"], "type": m["type"],
                          "seed": seed, "split": split, "n_params": m.get("n_params"),
                          "ms_per_sample_eval": m["score_timings"].get(split, {}).get("ms_per_sample")})
                rows.append(r)
    return pd.DataFrame(rows)


def summarise(long: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in long.columns if c in BOTH_CLASS_METRICS or c.startswith("tpr@")]
    g = long.groupby(["variant", "split"])
    mean = g[metric_cols].mean().add_suffix("_mean")
    std = g[metric_cols].std().fillna(0).add_suffix("_std")
    extra = g.agg(n_seeds=("seed", "nunique"), detector=("detector", "first"), type=("type", "first"),
                  n_params=("n_params", "first"), n=("n", "first"))
    return pd.concat([extra, mean, std], axis=1).reset_index()


def load_latency(res_dir: Path) -> pd.DataFrame:
    rows = []
    for p in glob.glob(str(res_dir / "latency" / "*.json")):
        d = load_json(p)
        r = {"detector": d["detector"], "device": d["device"], "n_params_lat": d.get("n_params"),
             "peak_mem_gb": d.get("peak_mem_gb"), "hw": d["env"].get("gpu") if d["device"] == "cuda" else d["env"].get("cpu_model")}
        for bs, b in d["batch"].items():
            r[f"bs{bs}_p50"] = b["ms_per_call_p50"]
            r[f"bs{bs}_p95"] = b["ms_per_call_p95"]
            r[f"bs{bs}_ms_per_sample"] = b["ms_per_sample"]
            r[f"bs{bs}_thru"] = b["throughput_per_s"]
        rows.append(r)
    return pd.DataFrame(rows)


def lat_lookup(lat: pd.DataFrame, det: str, device: str, col: str):
    if lat.empty:
        return None
    s = lat[(lat.detector == det) & (lat.device == device)]
    return None if s.empty or col not in s or pd.isna(s.iloc[0][col]) else float(s.iloc[0][col])


def family_of(t):
    return TYPE_FAMILY.get(t, "Reference")


def order_variants(summ: pd.DataFrame) -> list[str]:
    vs = summ.drop_duplicates("variant")[["variant", "type"]]
    vs = vs.assign(fo=vs.type.map(lambda t: FAMILIES[family_of(t)][0]), cal=vs.variant.str.endswith("@0.5"))
    return vs.sort_values(["fo", "cal", "variant"]).variant.tolist()


# =============================================================================  LaTeX tables
def tab_data(man: dict) -> str:
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Split & Sources & $N$ & \% inj. & Avg.\ chars & Role \\", r"\midrule"]
    role = lambda k: ("train" if k == "train" else "calibration" if k == "val" else "ID eval" if k == "test_id"
                      else "OOD eval" if k.startswith("ood") else "stress" if k.startswith("stress") else "over-defense")
    for k in ["train", "val", "test_id"] + sorted(x for x in man["splits"] if x not in ("train", "val", "test_id")):
        if k not in man["splits"]:
            continue
        s = man["splits"][k]
        srcs = ", ".join(sorted(s["sources"])).replace("_", r"\_")
        pct = 100 * s["n_pos"] / max(1, s["n"])
        lines.append(f"{SPLIT_DISPLAY.get(k, k).replace('_', ' ')} & \\footnotesize {srcs} & {s['n']:,} & {pct:.1f} & {s['mean_chars']:.0f} & {role(k)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def tab_main(summ, lat, cis, order, low_fpr) -> str:
    t = summ[summ.split == "test_id"].set_index("variant")
    tcol = f"tpr@{low_fpr:g}fpr"
    lines = [r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
             r" & & \multicolumn{5}{c}{ID test (threshold from validation)} & \multicolumn{3}{c}{Latency} \\",
             r"\cmidrule(lr){3-7}\cmidrule(lr){8-10}",
             r"Detector & Params & F1 & AUROC & AUPRC & FPR & TPR@1\% & GPU ms & GPU/s & CPU ms \\",
             r"\midrule"]
    fam_prev = None
    best = {c: t[f"{c}_mean"].max() for c in ["f1", "auroc", "auprc"] if f"{c}_mean" in t}
    best_fpr = t["fpr_mean"].min() if "fpr_mean" in t else None
    for v in order:
        if v not in t.index:
            continue
        r = t.loc[v]
        fam = family_of(r["type"])
        if fam_prev is not None and fam != fam_prev:
            lines.append(r"\midrule")
        fam_prev = fam

        def cell(c, nd=3, lower_better=False):
            mu = r.get(f"{c}_mean", np.nan)
            sd = r.get(f"{c}_std", 0.0)
            s = fmt(mu, nd)
            if r["n_seeds"] > 1 and not np.isnan(mu):
                s += f"$_{{\\pm {sd:.3f}}}$"
            is_best = (mu == best_fpr) if lower_better else (c in best and mu == best[c])
            return f"\\textbf{{{s}}}" if is_best and not np.isnan(mu) else s

        det = r["detector"]
        p = r["n_params"]
        ptxt = "--" if p is None or pd.isna(p) else (f"{p/1e9:.1f}B" if p >= 1e9 else f"{p/1e6:.0f}M" if p >= 1e6 else f"{p/1e3:.0f}K")
        lines.append(" & ".join([
            disp(v), ptxt, cell("f1"), cell("auroc"), cell("auprc"), cell("fpr", lower_better=True), cell(tcol),
            fmt_ms(lat_lookup(lat, det, "cuda", "bs1_p50")),
            fmt_ms(lat_lookup(lat, det, "cuda", "bs32_thru")),
            fmt_ms(lat_lookup(lat, det, "cpu", "bs1_p50"))]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def tab_ood(summ, order, splits) -> str:
    two = [s for s in splits if s.startswith(("ood", "stress"))]
    benign = [s for s in splits if s.startswith("benign")]
    cols = "l" + "r" * (len(two) + len(benign))
    head = ["Detector"] + [sdisp(s) for s in two] + [sdisp(s) for s in benign]
    lines = [f"\\begin{{tabular}}{{{cols}}}", r"\toprule"]
    if two and benign:
        lines.append(f" & \\multicolumn{{{len(two)}}}{{c}}{{AUROC $\\uparrow$}} & \\multicolumn{{{len(benign)}}}{{c}}{{FPR $\\downarrow$}} \\\\")
        lines.append(f"\\cmidrule(lr){{2-{1+len(two)}}}\\cmidrule(lr){{{2+len(two)}-{1+len(two)+len(benign)}}}")
    lines += [" & ".join(head) + r" \\", r"\midrule"]
    t = summ.set_index(["variant", "split"])
    best = {}
    for s in two:
        vals = [t.loc[(v, s), "auroc_mean"] for v in order if (v, s) in t.index and "@0.5" not in v]
        best[s] = np.nanmax(vals) if vals else None
    for s in benign:
        vals = [t.loc[(v, s), "fpr_mean"] for v in order if (v, s) in t.index and "@0.5" not in v]
        best[s] = np.nanmin(vals) if vals else None
    fam_prev = None
    for v in order:
        if "@0.5" in v:
            continue
        row = [disp(v)]
        typ = summ[summ.variant == v].type.iloc[0]
        fam = family_of(typ)
        if fam_prev is not None and fam != fam_prev:
            lines.append(r"\midrule")
        fam_prev = fam
        for s, mcol in [(s, "auroc_mean") for s in two] + [(s, "fpr_mean") for s in benign]:
            val = t.loc[(v, s), mcol] if (v, s) in t.index and mcol in t.columns else np.nan
            c = fmt(val)
            row.append(f"\\textbf{{{c}}}" if best.get(s) is not None and val == best[s] else c)
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def tab_judge(summ) -> str:
    j = summ[summ.type.isin(["llm_judge", "hf_classifier"])]
    if j.empty:
        return "% no zero-shot detectors were run"
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Detector & Threshold & ID F1 & ID recall & ID FPR & NotInject FPR \\", r"\midrule"]
    t = summ.set_index(["variant", "split"])
    for base in sorted(set(j.variant.str.replace("@0.5", "", regex=False))):
        for v, lab in [(base + "@0.5", "0.5 (default)"), (base, "val-calibrated")]:
            if (v, "test_id") not in t.index:
                continue
            r = t.loc[(v, "test_id")]
            ni = t.loc[(v, "benign_notinject"), "fpr_mean"] if (v, "benign_notinject") in t.index else np.nan
            lines.append(f"{disp(base)} & {lab} & {fmt(r['f1_mean'])} & {fmt(r['recall_mean'])} & {fmt(r['fpr_mean'])} & {fmt(ni)} \\\\")
        lines.append(r"\addlinespace")
    lines[-1:] = [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def tab_sig(sig: pd.DataFrame, best: str) -> str:
    if sig.empty:
        return "% no significance tests"
    lines = [r"\begin{tabular}{lrr}", r"\toprule",
             f"Compared with {disp(best)} & $\\Delta$F1 & $p$ \\\\", r"\midrule"]
    for _, r in sig.iterrows():
        lines.append(f"{disp(r['other'])} & {r['delta_f1']:+.3f} & {r['p']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# =============================================================================  figures
def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
        "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "lines.linewidth": 1.6,
        "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })
    return plt


def _clean(label: str) -> str:
    return label.replace("$^\\dagger$", "†").replace("\\_", "_")


def fig_roc(scores, metas, fig_dir, order):
    plt = _style()
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    seen_fam = {}  # line style distinguishes members of the same family
    for m in sorted(metas, key=lambda m: order.index(m["variant"]) if m["variant"] in order else 99):
        if m["seed"] != min(x["seed"] for x in metas if x["variant"] == m["variant"]):
            continue
        df = scores.get((m["variant"], m["seed"], "test_id"))
        if df is None or df.label.nunique() < 2:
            continue
        fam = family_of(m["type"])
        _, col, mk = FAMILIES[fam]
        k = seen_fam.get(fam, 0)
        seen_fam[fam] = k + 1
        fpr, tpr, _ = roc_curve(df.label, df.score)
        ax.plot(fpr, tpr, color=col, ls=["-", "--", ":", "-."][k % 4], label=_clean(disp(m["variant"])))
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate (log)")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=1, fontsize=6.5)
    fig.savefig(fig_dir / "roc_test_id.pdf")
    fig.savefig(fig_dir / "roc_test_id.png")
    plt.close(fig)


def _place_labels(ax, pts):
    """Greedy label placement avoiding other labels, markers and the axes border."""
    from matplotlib.transforms import Bbox

    fig = ax.figure
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(renderer=rnd)
    marks = []
    for x, y, _ in pts:
        px, py = ax.transData.transform((x, y))
        marks.append(Bbox([[px - 5, py - 5], [px + 5, py + 5]]))
    placed = []
    cands = [(dx, dy) for dy in (2, -10, 10, -18, 18, -26, 26, -34) for dx in (6, -6)]
    for x, y, lab in sorted(pts, key=lambda p: (-p[1], p[0])):
        for dx, dy in cands:
            t = ax.annotate(lab, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=6,
                            color="#52514e", ha="left" if dx > 0 else "right", va="bottom")
            bb = t.get_window_extent(renderer=rnd)
            ok = axbb.x0 <= bb.x0 and bb.x1 <= axbb.x1 and axbb.y0 <= bb.y0 and bb.y1 <= axbb.y1
            ok = ok and not any(bb.overlaps(o) for o in placed + marks)
            if ok:
                placed.append(bb)
                break
            t.remove()
        else:
            t = ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 2), fontsize=6, color="#52514e")
            placed.append(t.get_window_extent(renderer=rnd))


def fig_pareto(summ, lat, fig_dir, order):
    plt = _style()
    t = summ[(summ.split == "test_id") & (~summ.variant.str.endswith("@0.5"))]
    devs = [(d, n) for d, n in [("cuda", "GPU, batch size 1"), ("cpu", "CPU, batch size 1")]
            if not lat.empty and (lat.device == d).any()]
    if not devs:
        return
    fig, axes = plt.subplots(1, len(devs), figsize=(3.6 * len(devs), 3.0), sharey=True, squeeze=False)
    for ax, (dev, title) in zip(axes[0], devs):
        pts = []
        for _, r in t.iterrows():
            x = lat_lookup(lat, r.detector, dev, "bs1_p50")
            if x is None:
                continue
            _, col, mk = FAMILIES[family_of(r.type)]
            ax.scatter(x, r.f1_mean, s=36, color=col, marker=mk, edgecolor="white", linewidth=1.0, zorder=3)
            pts.append((x, r.f1_mean, _clean(disp(r.variant))))
        front, best = [], -1
        for x, y, _ in sorted(pts):
            if y > best:
                front.append((x, y))
                best = y
        if len(front) > 1:
            ax.plot(*zip(*front), color="#52514e", lw=0.8, ls="--", zorder=2)
        ax.set_xscale("log")
        if pts:
            xs = [p[0] for p in pts]
            ax.set_xlim(min(xs) / 1.5, max(xs) * 4)   # room for labels on the right
        ax.set_title(title)
        ax.set_xlabel("p50 latency per request (ms, log)")
        _place_labels(ax, pts)
    axes[0][0].set_ylabel("F1 on ID test")
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", markersize=6, label=f)
               for f, (_, c, m) in sorted(FAMILIES.items(), key=lambda x: x[1][0])
               if any(family_of(tt) == f for tt in t.type)]
    fig.legend(handles=handles, loc="upper center", ncol=min(6, len(handles)), frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    fig.savefig(fig_dir / "pareto_latency.pdf")
    fig.savefig(fig_dir / "pareto_latency.png")
    plt.close(fig)


def fig_fpr_benign(summ, fig_dir, order):
    plt = _style()
    benign = sorted((s for s in summ.split.unique() if s.startswith("benign")), reverse=True) + ["test_id"]
    vs = [v for v in order if "@0.5" not in v]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    width = 0.8 / len(benign)
    seq = ["#1c4f8f", "#2a78d6", "#8fb8ea"]  # one hue, dark -> light (sequential, ordered by split)
    t = summ.set_index(["variant", "split"])
    for j, s in enumerate(benign):
        vals = [t.loc[(v, s), "fpr_mean"] if (v, s) in t.index else np.nan for v in vs]
        ax.bar(np.arange(len(vs)) + (j - (len(benign) - 1) / 2) * width, vals, width * 0.92,
               color=seq[j % 3], label=SPLIT_DISPLAY.get(s, s) + (" (benign half)" if s == "test_id" else ""), edgecolor="white", linewidth=0.8)
    ax.set_xticks(np.arange(len(vs)))
    ax.set_xticklabels([_clean(disp(v)) for v in vs], rotation=30, ha="right", fontsize=6.5)
    ax.set_ylabel("False positive rate")
    ax.legend(frameon=False, ncol=len(benign), loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_ylim(0, 1.0)
    ax.grid(axis="x", visible=False)
    fig.savefig(fig_dir / "fpr_benign.pdf")
    fig.savefig(fig_dir / "fpr_benign.png")
    plt.close(fig)


def fig_heatmap(summ, fig_dir, order):
    plt = _style()
    splits = ["test_id"] + sorted(s for s in summ.split.unique() if s.startswith(("ood", "stress")))
    vs = [v for v in order if "@0.5" not in v]
    t = summ.set_index(["variant", "split"])
    M = np.array([[t.loc[(v, s), "auroc_mean"] if (v, s) in t.index else np.nan for s in splits] for v in vs])
    fig, ax = plt.subplots(figsize=(3.6, 0.28 * len(vs) + 0.9))
    im = ax.imshow(M, cmap="Blues", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels([SPLIT_DISPLAY.get(s, s) for s in splits], rotation=30, ha="right")
    ax.set_yticks(range(len(vs)))
    ax.set_yticklabels([_clean(disp(v)) for v in vs])
    ax.grid(False)
    for i in range(len(vs)):
        for j in range(len(splits)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if M[i, j] > 0.8 else "#0b0b0b")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="AUROC")
    fig.savefig(fig_dir / "ood_heatmap.pdf")
    fig.savefig(fig_dir / "ood_heatmap.png")
    plt.close(fig)



# =============================================================================  cascade
def cascade_analysis(scores, metas, cfg, lat):
    """Cheap detector decides confident cases; the band of inputs closest to its threshold
    (fraction q of validation data) is escalated to an expensive judge."""
    cc = cfg["eval"].get("cascade", {})
    cheap, exp = cc.get("cheap", "distilbert"), cc.get("expensive", "judge_qwen7b")
    fr = cc.get("fractions", [0.0, 0.05, 0.1, 0.2, 0.3, 1.0])
    seed_c = min((m["seed"] for m in metas if m["variant"] == cheap), default=None)
    seed_e = min((m["seed"] for m in metas if m["variant"] == exp), default=None)
    if seed_c is None or seed_e is None:
        return pd.DataFrame()
    thr = {m["variant"]: m["threshold"] for m in metas if (m["variant"], m["seed"]) in ((cheap, seed_c), (exp, seed_e))}
    vc = scores[(cheap, seed_c, "val")]
    dist_val = np.abs(vc.score.values - thr[cheap])
    dev = "cuda" if lat_lookup(lat, exp, "cuda", "bs1_p50") is not None else "cpu"
    lc = lat_lookup(lat, cheap, dev, "bs1_p50") or np.nan
    le = lat_lookup(lat, exp, dev, "bs1_p50") or np.nan
    rows = []
    for split in ["test_id"] + sorted(s for (v, sd, s) in scores if v == cheap and s.startswith("ood")):
        if (cheap, seed_c, split) not in scores or (exp, seed_e, split) not in scores:
            continue
        c, e = scores[(cheap, seed_c, split)], scores[(exp, seed_e, split)]
        if len(c) != len(e):
            continue
        for q in fr:
            band = np.quantile(dist_val, q) if 0 < q < 1 else (np.inf if q >= 1 else -1)
            esc = np.abs(c.score.values - thr[cheap]) <= band
            pred = np.where(esc, e.score.values >= thr[exp], c.score.values >= thr[cheap]).astype(float)
            m = binary_metrics(c.label.values, pred, 0.5)
            rows.append({"split": split, "frac_target": q, "frac_escalated": float(esc.mean()),
                         "f1": m.get("f1"), "fpr": m.get("fpr"), "recall": m.get("recall"),
                         "exp_latency_ms": lc + esc.mean() * le if q < 1 else le, "latency_device": dev,
                         "cheap": cheap, "expensive": exp})
    return pd.DataFrame(rows)


def tab_cascade(cas: pd.DataFrame) -> str:
    if cas.empty:
        return "% cascade analysis unavailable (needs both cheap and expensive detector runs)"
    t = cas[cas.split == "test_id"]
    ood = cas[cas.split != "test_id"].groupby("frac_target").f1.mean()
    lines = [r"\begin{tabular}{rrrrrr}", r"\toprule",
             r"Escalated (target) & Escalated (actual) & ID F1 & ID FPR & Mean OOD F1 & E[GPU ms] \\", r"\midrule"]
    for _, r in t.iterrows():
        q = r.frac_target
        lab = "0\\% (cheap only)" if q == 0 else "100\\% (judge only)" if q >= 1 else f"{100*q:.0f}\\%"
        lines.append(f"{lab} & {100*r.frac_escalated:.1f}\\% & {fmt(r.f1)} & {fmt(r.fpr)} & {fmt(ood.get(q, np.nan))} & {fmt_ms(r.exp_latency_ms)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# =============================================================================  macros / findings
def macros(summ, lat, man, cis, best_variant, sig, cas=None) -> str:
    out = ["% AUTO-GENERATED by scripts/aggregate.py -- do not edit by hand",
           "% Usage in text: \\res{<variant>}{<split>}{<metric>}, e.g. \\res{distilbert}{test-id}{f1}"]

    def put(a, b, c, val):
        out.append(f"\\expandafter\\def\\csname res@{key(a)}@{key(b)}@{key(c)}\\endcsname{{{val}}}")

    for _, r in summ.iterrows():
        for c in [c for c in summ.columns if c.endswith("_mean")]:
            if not pd.isna(r[c]):
                put(r.variant, r.split, c[:-5], fmt(r[c]))
                if c[:-5] in ("fpr", "recall", "f1", "accuracy"):
                    put(r.variant, r.split, c[:-5] + "-pct", f"{100 * r[c]:.1f}")
    for _, r in lat.iterrows():
        for c in r.index:
            if c.startswith("bs") and not pd.isna(r[c]):
                put(r.detector, "lat", f"{r.device}-{c}", fmt_ms(r[c]))
    for k, s in man["splits"].items():
        put("data", k, "n", f"{s['n']:,}".replace(",", "{,}"))
        put("data", k, "pos-pct", f"{100 * s['n_pos'] / max(1, s['n']):.1f}")
    if cas is not None and not cas.empty:
        for _, r in cas.iterrows():
            tag = f"q{int(round(100 * r.frac_target))}"
            put("cascade", r.split, f"{tag}-f1", fmt(r.f1))
            put("cascade", r.split, f"{tag}-fpr", fmt(r.fpr))
            put("cascade", r.split, f"{tag}-esc-pct", f"{100 * r.frac_escalated:.1f}")
            put("cascade", r.split, f"{tag}-ms", fmt_ms(r.exp_latency_ms))
    for v, (lo, hi) in cis.items():
        put(v, "test-id", "f1-ci", f"[{lo:.3f}, {hi:.3f}]")
    out.append(f"\\def\\bestvariant{{{disp(best_variant) if best_variant else '--'}}}")
    hw = lambda dev: (lat[lat.device == dev].hw.dropna().iloc[0] if not lat.empty and (lat.device == dev).any()
                      and lat[lat.device == dev].hw.notna().any() else "--")
    out.append(f"\\def\\gpuname{{{str(hw('cuda')).replace('_', ' ')}}}")
    out.append(f"\\def\\cpuname{{{str(hw('cpu')).replace('_', ' ').replace('(R)', '').replace('(TM)', '')}}}")
    return "\n".join(out) + "\n"


def findings_md(summ, lat, best_variant, sig) -> str:
    t = summ[~summ.variant.str.endswith("@0.5")]
    L = ["# Auto-generated findings (verify every claim in the paper against this)\n"]

    def rank(split, metric, asc=False, k=20):
        s = t[t.split == split].sort_values(f"{metric}_mean", ascending=asc)
        return "\n".join(f"  {i+1}. {r.variant}: {r[f'{metric}_mean']:.4f} (±{r[f'{metric}_std']:.4f}, seeds={r.n_seeds})"
                         for i, (_, r) in enumerate(s.head(k).iterrows()) if not pd.isna(r[f"{metric}_mean"]))

    L += ["## ID test F1", rank("test_id", "f1"), "", "## ID test AUROC", rank("test_id", "auroc"), ""]
    for s in sorted(t.split.unique()):
        if s.startswith(("ood", "stress")):
            L += [f"## {s} AUROC", rank(s, "auroc"), ""]
        if s.startswith("benign"):
            L += [f"## {s} FPR (lower is better)", rank(s, "fpr", asc=True), ""]
    ood = t[t.split.str.startswith("ood")].groupby("variant").auroc_mean.mean().sort_values(ascending=False)
    L += ["## Mean OOD AUROC", "\n".join(f"  {v}: {x:.4f}" for v, x in ood.items()), ""]
    if not lat.empty:
        L += ["## Latency (p50 ms, bs=1)"]
        for _, r in lat.sort_values(["device", "bs1_p50"]).iterrows():
            L.append(f"  [{r.device}] {r.detector}: p50={r.get('bs1_p50', np.nan):.2f} ms, "
                     f"bs32 throughput={r.get('bs32_thru', np.nan):.1f}/s")
        L.append("")
    if not sig.empty:
        L += [f"## Paired bootstrap vs best ({best_variant}) on ID F1",
              "\n".join(f"  {r.other}: dF1={r.delta_f1:+.4f} p={r.p:.4f}" for _, r in sig.iterrows()), ""]
    return "\n".join(L)


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--paper-dir", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    res = Path(cfg["paths"]["results_dir"])
    paper = Path(args.paper_dir) if args.paper_dir else Path(cfg["paths"]["root"]) / "paper"
    gen, figd, tabd = ensure_dir(paper / "generated"), ensure_dir(paper / "figures"), ensure_dir(res / "tables")
    low = cfg["eval"].get("low_fpr", 0.01)

    man = load_json(Path(cfg["paths"]["data_dir"]) / "processed" / "manifest.json")
    metas, scores = load_runs(res, cfg)
    if not metas:
        raise SystemExit("no runs found under results/runs")
    long = compute_long(metas, scores, low)
    long.to_csv(tabd / "metrics_long.csv", index=False)
    summ = summarise(long)
    summ.to_csv(tabd / "metrics_summary.csv", index=False)
    lat = load_latency(res)
    lat.to_csv(tabd / "latency.csv", index=False)
    order = order_variants(summ)

    # bootstrap CIs (first seed of each variant) + paired significance vs best ID-F1 variant
    n_boot = cfg["eval"].get("bootstrap", 1000)
    first_seed = {m["variant"]: min(x["seed"] for x in metas if x["variant"] == m["variant"]) for m in metas}
    thr = {m["variant"]: m["threshold"] for m in metas if m["seed"] == first_seed[m["variant"]]}
    cis = {}
    for v, s in first_seed.items():
        df = scores.get((v, s, "test_id"))
        if df is not None and df.label.nunique() == 2:
            cis[v] = bootstrap_ci(df.label.values, df.score.values, thr[v], "f1", n_boot)
    t = summ[(summ.split == "test_id") & (~summ.variant.str.endswith("@0.5"))]
    best_variant = t.sort_values("f1_mean", ascending=False).variant.iloc[0] if len(t) else None
    sig_rows = []
    if best_variant:
        db = scores[(best_variant, first_seed[best_variant], "test_id")]
        for v in order:
            if v == best_variant or v.endswith("@0.5") or (v, first_seed.get(v), "test_id") not in scores:
                continue
            do = scores[(v, first_seed[v], "test_id")]
            if len(do) != len(db) or not np.array_equal(do.label.values, db.label.values):
                continue
            p = paired_bootstrap_pvalue(db.label.values, db.score.values, do.score.values,
                                        thr[best_variant], thr[v], "f1", n_boot)
            d = (binary_metrics(db.label.values, db.score.values, thr[best_variant])["f1"]
                 - binary_metrics(do.label.values, do.score.values, thr[v])["f1"])
            sig_rows.append({"other": v, "delta_f1": d, "p": p})
    sig = pd.DataFrame(sig_rows)
    sig.to_csv(tabd / "significance.csv", index=False)
    pd.DataFrame([{"variant": k, "f1_ci_lo": a, "f1_ci_hi": b} for k, (a, b) in cis.items()]).to_csv(
        tabd / "bootstrap_ci.csv", index=False)

    cas = cascade_analysis(scores, metas, cfg, lat)
    cas.to_csv(tabd / "cascade.csv", index=False)
    splits = sorted(summ.split.unique())
    (gen / "tab_data.tex").write_text(tab_data(man))
    (gen / "tab_main.tex").write_text(tab_main(summ, lat, cis, order, low))
    (gen / "tab_ood.tex").write_text(tab_ood(summ, order, splits))
    (gen / "tab_judge.tex").write_text(tab_judge(summ))
    (gen / "tab_sig.tex").write_text(tab_sig(sig, best_variant))
    (gen / "tab_cascade.tex").write_text(tab_cascade(cas))
    (gen / "macros.tex").write_text(macros(summ, lat, man, cis, best_variant, sig, cas))
    fnd = findings_md(summ, lat, best_variant, sig)
    if not cas.empty:
        fnd += "\n## Cascade (cheap -> judge)\n" + cas.to_string(index=False) + "\n"
    (res / "findings.md").write_text(fnd)

    fig_roc(scores, metas, figd, order)
    fig_pareto(summ, lat, figd, order)
    fig_fpr_benign(summ, figd, order)
    fig_heatmap(summ, figd, order)
    LOG.info("tables -> %s | latex -> %s | figures -> %s", tabd, gen, figd)
    print((res / "findings.md").read_text())


if __name__ == "__main__":
    main()
