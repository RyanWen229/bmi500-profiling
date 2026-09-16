#!/usr/bin/env python
"""Profile the Scanpy PBMC workflow with coarse section-level timings.

The rank-gene step is deliberately executed once, under ``cProfile``. This
avoids adding an unprofiled duplicate computation to the reported runtime.
"""

import argparse
import cProfile
import json
import os
from pathlib import Path
import time

import scanpy as sc


def parse_args():
    parser = argparse.ArgumentParser(description="Run and profile a PBMC Scanpy workflow.")
    parser.add_argument("--data-dir", default="data", help="Directory containing PBMC folders")
    parser.add_argument("--data-set", default="pbmc3k", choices=("pbmc3k", "pbmc6k", "pbmc10k"))
    parser.add_argument("--out-dir", default="results", help="Directory for AnnData outputs")
    parser.add_argument("--profile-dir", default="profiles", help="Directory for cProfile outputs")
    parser.add_argument("--num-threads", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[variable] = str(args.num_threads)

    sc.settings.verbosity = 2
    sc.settings.n_jobs = args.num_threads
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.profile_dir).mkdir(parents=True, exist_ok=True)
    timings = {}

    def section(name, function):
        start = time.perf_counter()
        result = function()
        elapsed = time.perf_counter() - start
        timings[name] = elapsed
        print(f"SECTION_TIMING dataset={args.data_set} section={name} seconds={elapsed:.4f}", flush=True)
        return result

    print(f"RUN_METADATA dataset={args.data_set} threads={args.num_threads}", flush=True)
    input_dir = Path(args.data_dir) / args.data_set / "filtered_gene_bc_matrices"
    # Disable Scanpy's on-disk read cache so every dataset measurement includes
    # comparable Matrix Market input I/O rather than a cache hit from a prior run.
    adata = section("read_10x", lambda: sc.read_10x_mtx(input_dir, var_names="gene_symbols", cache=False))
    adata.var_names_make_unique()
    print(f"DATASET_SHAPE dataset={args.data_set} cells={adata.n_obs} genes={adata.n_vars}", flush=True)

    section("filter", lambda: (sc.pp.filter_cells(adata, min_genes=200), sc.pp.filter_genes(adata, min_cells=3)))
    section("normalize_log1p", lambda: (sc.pp.normalize_total(adata, target_sum=1e4), sc.pp.log1p(adata)))
    def select_hvg():
        nonlocal adata
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
        adata.raw = adata
        adata = adata[:, adata.var.highly_variable]

    section("highly_variable_genes", select_hvg)

    section("scale", lambda: sc.pp.scale(adata))
    section("pca", lambda: sc.tl.pca(adata, svd_solver="arpack", n_comps=30))
    section("neighbors", lambda: sc.pp.neighbors(adata, n_pcs=30))
    section("louvain", lambda: sc.tl.louvain(adata, resolution=0.5))
    section("umap", lambda: sc.tl.umap(adata, n_components=30))
    section("write_h5ad", lambda: adata.write(Path(args.out_dir) / f"{args.data_set}.scanpy.h5ad"))

    profile_path = Path(args.profile_dir) / f"rank_genes_{args.data_set}_profile.prof"

    def rank_genes():
        profiler = cProfile.Profile()
        profiler.enable()
        sc.tl.rank_genes_groups(adata, "louvain", method="wilcoxon", use_raw=True)
        profiler.disable()
        profiler.dump_stats(profile_path)

    section("rank_gene_groups", rank_genes)
    print("TIMINGS_JSON " + json.dumps({"dataset": args.data_set, "sections": timings}, sort_keys=True), flush=True)
    print(f"PROFILE_OUTPUT {profile_path}", flush=True)


if __name__ == "__main__":
    main()
