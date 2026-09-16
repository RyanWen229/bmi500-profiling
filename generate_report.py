#!/usr/bin/env python
"""Build figures and a self-contained PDF from PBMC profiling artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import pstats
import re
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASETS = ("pbmc3k", "pbmc6k", "pbmc10k")
SECTIONS = (
    "read_10x", "filter", "normalize_log1p", "highly_variable_genes", "scale",
    "pca", "neighbors", "louvain", "umap", "write_h5ad", "rank_gene_groups",
)


def elapsed_seconds(value: str) -> float:
    """Parse GNU time's m:ss.xx or h:mm:ss.xx elapsed-time field."""
    pieces = [float(part) for part in value.strip().split(":")]
    if len(pieces) == 2:
        return 60 * pieces[0] + pieces[1]
    return 3600 * pieces[0] + 60 * pieces[1] + pieces[2]


def parse_log(dataset: str) -> dict:
    text = (ROOT / "logs" / f"{dataset}.console.log").read_text()
    shape = re.search(r"DATASET_SHAPE dataset=\S+ cells=(\d+) genes=(\d+)", text)
    timing = re.search(r"TIMINGS_JSON (\{.*\})", text)
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.+)", text)
    cpu = re.search(r"Percent of CPU this job got: (\d+)%", text)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    user = re.search(r"User time \(seconds\): ([\d.]+)", text)
    system = re.search(r"System time \(seconds\): ([\d.]+)", text)
    if not all((shape, timing, wall, cpu, rss, user, system)):
        raise ValueError(f"Could not fully parse logs for {dataset}")
    return {
        "dataset": dataset,
        "cells": int(shape.group(1)),
        "genes": int(shape.group(2)),
        "sections": json.loads(timing.group(1))["sections"],
        "wall_s": elapsed_seconds(wall.group(1)),
        "cpu_pct": int(cpu.group(1)),
        "max_rss_gib": int(rss.group(1)) / 1024**2,
        "user_s": float(user.group(1)),
        "system_s": float(system.group(1)),
    }


def text_page(pdf: PdfPages, title: str, body: str, size: float = 10, mono: bool = False) -> None:
    width = 112 if mono else 105
    wrapped = []
    for line in body.splitlines():
        if not line:
            wrapped.append("")
        else:
            wrapped.extend(wrap(line, width=width, replace_whitespace=False, break_on_hyphens=False))
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.07, 0.955, title, fontsize=17, fontweight="bold", va="top")
    fig.text(0.07, 0.915, "\n".join(wrapped), fontsize=size, family="monospace" if mono else "sans-serif", va="top", linespacing=1.35)
    pdf.savefig(fig)
    plt.close(fig)


def profile_top(dataset: str, limit: int = 7):
    stats = pstats.Stats(str(ROOT / "profiles" / f"rank_genes_{dataset}_profile.prof")).stats
    values = []
    for (filename, line, function), (_, _, tottime, cumulative, _) in stats.items():
        if function in {"<method 'disable' of '_lsprof.Profiler' objects>", "fn_compatible", "rank_genes_groups", "compute_statistics"}:
            continue
        label = f"{Path(filename).name}:{function}"
        values.append((tottime, cumulative, label))
    return sorted(values, reverse=True)[:limit]


def console_pages(pdf: PdfPages, dataset: str) -> None:
    raw_lines = (ROOT / "logs" / f"{dataset}.console.log").read_text().splitlines()
    lines = [piece for line in raw_lines for piece in (wrap(line, width=112, replace_whitespace=False, break_on_hyphens=False) or [""])]
    per_page = 57
    for start in range(0, len(lines), per_page):
        end = min(start + per_page, len(lines))
        text_page(pdf, f"Appendix: {dataset} console output ({start + 1}–{end} of {len(lines)})", "\n".join(lines[start:end]), size=5.7, mono=True)


def main() -> None:
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    records = [parse_log(dataset) for dataset in DATASETS]
    cells = np.array([record["cells"] for record in records])
    runtime = pd.DataFrame([record["sections"] for record in records], index=DATASETS)[list(SECTIONS)]
    resources = pd.DataFrame([
        {"Dataset": r["dataset"], "Input cells": r["cells"], "Genes": r["genes"],
         "Wall (s)": r["wall_s"], "CPU (%)": r["cpu_pct"], "Peak RSS (GiB)": r["max_rss_gib"]}
        for r in records
    ])
    forecast = pd.DataFrame(index=SECTIONS)
    forecast["20K projected seconds"] = [max(0, np.polyfit(cells, runtime[section].to_numpy(), 1)[0] * 20000 + np.polyfit(cells, runtime[section].to_numpy(), 1)[1]) for section in SECTIONS]
    forecast["20K projected seconds"] = forecast["20K projected seconds"].round(1)
    forecast.loc["TOTAL (instrumented sections)"] = forecast.sum()
    forecast.to_csv(ROOT / "results" / "pbmc_20k_section_projection.csv", index_label="section")
    runtime.round(4).to_csv(ROOT / "results" / "pbmc_section_timings.csv", index_label="dataset")
    resources.round(3).to_csv(ROOT / "results" / "pbmc_resource_summary.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    labels = [f"{d}\n({c:,} cells)" for d, c in zip(DATASETS, cells)]
    fig, ax = plt.subplots(figsize=(12, 7))
    bottom = np.zeros(len(DATASETS))
    colors = plt.cm.tab20(np.linspace(0, 1, len(SECTIONS)))
    for section, color in zip(SECTIONS, colors):
        values = runtime[section].to_numpy()
        ax.bar(labels, values, bottom=bottom, label=section, color=color, edgecolor="white", linewidth=.4)
        bottom += values
    ax.set_title("Instrumented Scanpy section runtimes vs. dataset size")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_xlabel("Dataset (raw input cells)")
    ax.legend(title="Section", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "section_runtime_growth.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=False)
    for axis, dataset in zip(axes, DATASETS):
        top = profile_top(dataset)
        names = [item[2].replace("_", "_\u200b") for item in top][::-1]
        times = [item[0] for item in top][::-1]
        axis.barh(names, times, color="#4878A8")
        axis.set_title(dataset)
        axis.set_xlabel("Self time (s)")
        axis.tick_params(axis="y", labelsize=7)
    fig.suptitle("cProfile: rank_gene_groups top functions by self time", fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(figures / "rank_gene_groups_cprofile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    total_section = runtime.sum(axis=1)
    rank_share = runtime["rank_gene_groups"] / total_section * 100
    bottleneck = "neighbors" if runtime.loc["pbmc10k"].idxmax() == "neighbors" else runtime.loc["pbmc10k"].idxmax()
    summary = (
        "Scope and method\n"
        "Single-threaded Scanpy 1.10.2 workflow (PBMC Matrix Market input); one measured run per dataset. "
        "The reader cache is disabled so input timing is comparable. Each run uses /usr/bin/time -v for process-level "
        "resources and time.perf_counter around every workflow section. rank_gene_groups is executed exactly once, under cProfile.\n\n"
        "Key findings\n"
        f"• Total instrumented runtime grows from {total_section.iloc[0]:.1f}s (3K) to {total_section.iloc[2]:.1f}s (10K); external wall time is {resources.iloc[0]['Wall (s)']:.2f}s to {resources.iloc[2]['Wall (s)']:.2f}s.\n"
        f"• At 10K, {bottleneck} is the largest coarse section ({runtime.loc['pbmc10k', bottleneck]:.1f}s), followed by rank_gene_groups ({runtime.loc['pbmc10k', 'rank_gene_groups']:.1f}s) and UMAP ({runtime.loc['pbmc10k', 'umap']:.1f}s).\n"
        f"• Within rank_gene_groups, cProfile identifies pandas rank / Scanpy _ranks as dominant; sparse column indexing and basic-statistics work become substantial at 10K.\n"
        "• Peak RSS increases from 0.82 GiB (3K) to 1.66 GiB (10K), consistent with dense scaling and larger graph/embedding data.\n\n"
        "20K estimate\n"
        "The projection is an ordinary least-squares linear fit of the three observed points against raw cell count. It is a planning estimate, not a guarantee: neighbor graph construction and UMAP can scale nonlinearly, and cluster composition changes rank-gene work."
    )

    with PdfPages(ROOT / "PBMC_profiling_report.pdf") as pdf:
        text_page(pdf, "PBMC 3K / 6K / 10K Performance Profiling", summary, size=11)

        fig, ax = plt.subplots(figsize=(10.5, 5.5))
        ax.axis("off")
        table = ax.table(cellText=[[f"{value:.2f}" if isinstance(value, float) else value for value in row] for row in resources.values], colLabels=resources.columns, cellLoc="center", loc="center")
        table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.08, 1.8)
        ax.set_title("Coarse-grain wall time, CPU utilization, and memory", pad=20, fontsize=15, fontweight="bold")
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 7))
        image = plt.imread(figures / "section_runtime_growth.png")
        ax.imshow(image); ax.axis("off")
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 11))
        image = plt.imread(figures / "rank_gene_groups_cprofile.png")
        ax.imshow(image); ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        pdf.savefig(fig); plt.close(fig)

        projection_text = "20K projected runtimes (seconds)\n\n" + forecast.to_string() + "\n\nInterpretation: neighbors is the main projected cost (~" + str(forecast.loc["neighbors", "20K projected seconds"]) + "s). Rank-gene work (~" + str(forecast.loc["rank_gene_groups", "20K projected seconds"]) + "s) is the next major CPU hotspot. The fitted instrumented total is ~" + str(forecast.loc["TOTAL (instrumented sections)", "20K projected seconds"]) + "s; allow additional startup/import overhead and scheduling variability."
        text_page(pdf, "20K extrapolation and bottleneck assessment", projection_text, size=10, mono=True)

        slurm_text = "Submit commands\n\n  sbatch slurm/pbmc3k.sbatch\n  sbatch slurm/pbmc6k.sbatch\n  sbatch slurm/pbmc10k.sbatch\n\nBatch configuration\n\n  pbmc3k: 1 CPU, 16G, 01:00:00\n  pbmc6k: 1 CPU, 24G, 02:00:00\n  pbmc10k: 1 CPU, 32G, 04:00:00\n\nEach script changes to SLURM_SUBMIT_DIR and invokes scripts/run_pbmc_experiment.sh. The runner activates the bmi500 environment, constrains BLAS/OMP/Numba to the allocated CPU count, writes logs/<dataset>.console.log, results/<dataset>.scanpy.h5ad, and profiles/rank_genes_<dataset>_profile.prof.\n\nReproducibility note\n\nThe figure and PDF are regenerated with: python generate_report.py"
        text_page(pdf, "Slurm batch scripts and reproduction", slurm_text, size=10, mono=True)

        scripts = "\n\n".join(
            f"--- slurm/{dataset}.sbatch ---\n" + (ROOT / "slurm" / f"{dataset}.sbatch").read_text().strip()
            for dataset in DATASETS
        )
        text_page(pdf, "Appendix: Slurm batch-script listings", scripts, size=7.5, mono=True)

        for dataset in DATASETS:
            console_pages(pdf, dataset)

    print(f"Wrote {ROOT / 'PBMC_profiling_report.pdf'}")


if __name__ == "__main__":
    main()
