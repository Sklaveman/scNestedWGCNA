import os
import optuna
import warnings
import argparse
import numpy as np
import pandas as pd
from functools import partial
from joblib import Parallel, delayed, cpu_count

from src.utils import (
    get_cgms,
    NOISE_MODULE,
    get_fgms_from_cgm,
    suppress_warnings,
    save_modules_to_gmt,
    topk_coexpression_filter,
)

from src.parameter_optimization import (
    optimize_umap_params,
    optimize_cgm_clustering_params,
    _sizes_from_fracs,
)

from typing import Any, Dict, List, Optional, Tuple

# Only UserWarning/FutureWarning are silenced, not a blanket "ignore".
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "modules"
)


def _banner(title: str, width: int = 50) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def load_and_filter_data(input_path: str) -> pd.DataFrame:
    """Load the expression CSV and drop all-zero genes."""
    expr_df = pd.read_csv(input_path, index_col=0)
    zero_genes = expr_df.loc[:, (expr_df == 0).all()].columns
    return expr_df.loc[:, ~expr_df.columns.isin(zero_genes)]


def load_gene_network(path: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Load a gene-regulatory-network CSV (e.g. data/grns/grn_multisim_*.csv,
    columns: target, regulator, effect) for use as `gene_network` in
    knowledge_score optimization. `compute_knowledge_score()` reads the
    first two columns positionally, so no column-name handling is needed.

    knowledge_score is the only supported objective, so the CLI requires this
    file (`--gene-network-file`).
    """
    if path is None:
        return None
    return pd.read_csv(path)


def frac_range(start: float, stop: float, step: float) -> List[float]:
    """Inclusive float range, rounded to avoid classic np.arange fencepost
    drift (e.g. stop being excluded/included inconsistently near boundaries)."""
    n_steps = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 6) for i in range(n_steps) if start + i * step <= stop + 1e-9]


def get_umap_params(
    expr_df: pd.DataFrame,
    optimize_umap: bool,
    min_clust_size_umap: int,
    min_samples_umap: int,
    n_trials_umap: int,
    provided_umap_params: Optional[Dict] = None,
) -> Dict:
    """Optimize the UMAP parameters or validate the provided ones."""
    _banner("UMAP Parameter Configuration")

    if optimize_umap:
        print("Optimizing UMAP parameters...")
        best_params = optimize_umap_params(
            expr_df,
            min_clust_size=min_clust_size_umap,
            min_samples=min_samples_umap,
            n_trials=n_trials_umap,
        )
        print("\nOptimized UMAP parameters found:")
    else:
        print("Using provided UMAP parameters:")
        best_params = provided_umap_params

    for param, value in best_params.items():
        print(f"  {param}: {value}")

    return best_params


def get_cgm_params(
    expr_df: pd.DataFrame,
    best_umap_params: Dict,
    optimize_cgm: bool,
    cgm_min_clust_frac_lst: List[float],
    gene_network: Optional[pd.DataFrame],
    provided_cgm_params: Optional[Dict] = None,
) -> Dict:
    """
    CGM parameters via the cheap re-condensing sweep in
    `optimize_cgm_clustering_params` (knowledge_score only), with
    min_cluster_size expressed as fractions of total gene count rather than
    a fixed absolute grid.
    """
    _banner("CGM Parameter Configuration")

    if optimize_cgm:
        print("Optimizing CGM parameters (knowledge_score, cheap re-condensing sweep)...")
        min_clust_size_lst = _sizes_from_fracs(
            cgm_min_clust_frac_lst, expr_df.shape[1], 5
        )
        best_params = optimize_cgm_clustering_params(
            expr_df,
            best_umap_params=best_umap_params,
            gene_network=gene_network,
            min_clust_size_lst=min_clust_size_lst,
        )
        print("\nOptimized CGM parameters found:")
    else:
        print("Using provided CGM parameters:")
        best_params = provided_cgm_params

    for param, value in best_params.items():
        print(f"  {param}: {value}")

    return best_params


def run_cgm_clustering(
    expr_df: pd.DataFrame, cgm_params: Dict, cgm_noise_filter: str = "none"
) -> pd.DataFrame:
    """
    Returns just `modules_df`; the dissimilarity matrix, embedding, module
    counts and clusterer `get_cgms()` also produces are discarded.
    """
    _banner("CGM Clustering")

    _dissim, _embedding, modules_df, module_counts, _clusterer = get_cgms(
        expr_df,
        min_clust_size=cgm_params["min_clust_size"],
        min_samples=cgm_params["min_samples"],
        n_neighbors=cgm_params["n_neighbors"],
        n_components=cgm_params["n_components"],
        min_dist=cgm_params["min_dist"],
        noise_method="hdbscan" if cgm_noise_filter == "hdbscan" else "none",
    )

    n_noise = int((modules_df.module == NOISE_MODULE).sum())
    print(f"\nResults:")
    print(f"  Number of modules: {(module_counts.module != NOISE_MODULE).sum()}")
    if cgm_noise_filter == "hdbscan":
        print(f"  HDBSCAN-native noise genes: {n_noise} ({n_noise / len(modules_df):.1%})")
    print(f"  Module distribution:")
    print(module_counts)

    return modules_df


NORM_COLUMNS = ["CGM", "corr_thr", "k", "n_ings", "knowledge_score"]
_NAN_NORM = dict.fromkeys(["corr_thr", "k", "n_ings", "knowledge_score"], np.nan)


def _process_cgm(
    module: str,
    module_expr_df: pd.DataFrame,
    core_genes: List[str],
    best_umap_params: Dict,
    optimize_clust_params: bool,
    fgm_min_clust_frac_lst: List[float],
    merge_size_multiplier_lst: List[float],
    gene_network: Optional[pd.DataFrame],
    hdbscan_n_jobs: int,
    noise_method: str = "kme",
    fgm_noise_kwargs: Optional[Dict] = None,
    norm_kwargs: Optional[Dict] = None,
    return_embeddings: bool = False,
) -> Tuple[str, pd.DataFrame, Optional[str], Dict[str, Any]]:
    """
    Module-level (picklable) per-CGM worker, used both for sequential
    (n_jobs=1) and joblib-parallel (n_jobs>1) dispatch in `extract_all_fgms`.
    A failure on one CGM does not abort the run: that CGM's genes are kept as
    a single unrefined block.

    Returns `(module, submodules_df, error, norm_settings)` — `error` is `None`
    on success, or the exception message on failure, so `extract_all_fgms()`
    can report which CGMs (if any) fell back to the unrefined block. The
    unnormalized/normalized embeddings `get_fgms_from_cgm()` also returns are
    dropped unless `return_embeddings=True`; then they are appended to
    `submodules_df` as columns `raw_umap1..d` / `norm_umap1..d` (each CGM is
    its own UMAP fit, so coordinates are not comparable across CGMs). Costs one
    extra UMAP fit per CGM.
    """
    print(f"Processing CGM: {module}")
    try:
        submodules_df, embedding_raw, embedding_norm = get_fgms_from_cgm(
            module_expr_df,
            best_umap_params=best_umap_params,
            parent_module_name=module,
            optimize_clust_params=optimize_clust_params,
            min_clust_frac_lst=fgm_min_clust_frac_lst,
            merge_size_multiplier_lst=merge_size_multiplier_lst,
            gene_network=gene_network,
            core=core_genes,
            hdbscan_n_jobs=hdbscan_n_jobs,
            return_raw_embedding=return_embeddings,
            noise_method=noise_method,
            **(fgm_noise_kwargs or {}),
            **(norm_kwargs or {}),
        )
        # carried explicitly: attrs do not survive pickling / concat / join
        norm_settings = submodules_df.attrs.get("normalization", _NAN_NORM)
        if return_embeddings:
            submodules_df = submodules_df.join(
                [
                    pd.DataFrame(
                        emb,
                        index=submodules_df.index,
                        columns=[f"{prefix}_umap{i + 1}" for i in range(emb.shape[1])],
                    )
                    for prefix, emb in (("raw", embedding_raw), ("norm", embedding_norm))
                ]
            )
        print(f"Done: {module}\n")
        return module, submodules_df, None, norm_settings
    except Exception as e:
        print(f"Error during FGM extraction for {module}: {e}")
        fallback_df = pd.DataFrame(
            {
                "module": module,
                "module_core": module,
                "merged_module": module,
                "merged_module_core": module,
                "kme": np.nan,
                "kme_qval": np.nan,
            },
            index=module_expr_df.columns,
        )
        return module, fallback_df, f"{type(e).__name__}: {e}", _NAN_NORM


def extract_all_fgms(
    expr_df: pd.DataFrame,
    cgm_df: pd.DataFrame,
    best_umap_params: Dict[str, int],
    optimize_clust_params: bool,
    fgm_min_clust_frac_lst: List[float],
    merge_size_multiplier_lst: List[float],
    gene_network: Optional[pd.DataFrame] = None,
    n_jobs: int = 1,
    noise_method: str = "kme",
    fgm_noise_kwargs: Optional[Dict] = None,
    norm_kwargs: Optional[Dict] = None,
    return_embeddings: bool = False,
    extra_noise_genes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extract all FGMs from given CGMs. Returns just `fgm_df` (the per-CGM
    normalization settings are in `fgm_df.attrs["normalization"]`, stored as
    `DataFrame.to_dict()`: a DataFrame in `.attrs` breaks pandas, so rebuild it
    with `pd.DataFrame(attrs["normalization"], columns=NORM_COLUMNS)`).
    With `return_embeddings=True` it also has the per-CGM `raw_umap*` /
    `norm_umap*` columns (see `_process_cgm`; NaN for a CGM that fell back).

    `noise_method` (same name as in `get_cgms`) is the FGM-stage noise method,
    see `get_fgms_from_cgm`: `"hdbscan"` = hard HDBSCAN labels for the fine and
    the merged cut with their native `-1` as `module_-1`; `"kme"` = soft
    assignments plus the kME test on the merged cut. `fgm_noise_kwargs` carries
    only the kME settings `kme_fdr` / `kme_min` (used with `"kme"`).

    Noise genes get no FGMs but are kept: genes of the CGM noise set
    (`NOISE_MODULE`, `cgm_noise_filter="hdbscan"`) and `extra_noise_genes`
    (e.g. removed by the top-k pre-filter) are appended with `NOISE_MODULE` in
    all four label columns (kme / embeddings NaN).

    CGMs are processed via `_process_cgm()`, sorted by
    descending size first for better load balancing under joblib, and
    `hdbscan_n_jobs` (per-CGM internal HDBSCAN thread count) is throttled
    against `n_jobs` to avoid oversubscription: n_jobs=1 gets all cores for
    HDBSCAN internally, n_jobs>1 splits cores roughly evenly across workers.
    """
    if noise_method not in ("hdbscan", "kme"):
        raise ValueError(f"noise_method must be 'hdbscan' or 'kme', got {noise_method!r}")
    if fgm_noise_kwargs and "noise_method" in fgm_noise_kwargs:
        raise ValueError("pass noise_method as its own argument, not inside fgm_noise_kwargs")
    fgm_dict = {}
    is_cgm_noise = cgm_df.module == NOISE_MODULE
    noise_genes = cgm_df.index[is_cgm_noise].append(pd.Index([] if extra_noise_genes is None else extra_noise_genes))
    cgm_df = cgm_df[~is_cgm_noise].copy()

    modules = list(cgm_df.module.unique())
    module_sizes = cgm_df.module.value_counts().to_dict()
    modules.sort(key=lambda m: module_sizes[m], reverse=True)

    if n_jobs == 1:
        hdbscan_n_jobs = -1
    else:
        available = cpu_count()
        effective_workers = n_jobs if n_jobs > 0 else available
        hdbscan_n_jobs = max(1, available // max(1, effective_workers))

    members = cgm_df.groupby("module").groups
    core_members = cgm_df.groupby("core_module").groups
    tasks = [
        (
            module,
            expr_df[members[module]],
            core_members[module].to_list() if module in core_members else [],
        )
        for module in modules
    ]

    process = partial(
        _process_cgm,
        best_umap_params=best_umap_params,
        optimize_clust_params=optimize_clust_params,
        fgm_min_clust_frac_lst=fgm_min_clust_frac_lst,
        merge_size_multiplier_lst=merge_size_multiplier_lst,
        gene_network=gene_network,
        hdbscan_n_jobs=hdbscan_n_jobs,
        noise_method=noise_method,
        fgm_noise_kwargs=fgm_noise_kwargs,
        norm_kwargs=norm_kwargs,
        return_embeddings=return_embeddings,
    )
    if n_jobs == 1:
        results = [process(*task) for task in tasks]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process)(*task) for task in tasks
        )

    failed = []
    norm_rows = []
    for module, submodules_df, error, norm_settings in results:
        fgm_dict[module] = submodules_df
        norm_rows.append({"CGM": module, **norm_settings})
        if error is not None:
            failed.append(
                {"CGM": module, "n_genes": int(module_sizes[module]), "error": error}
            )

    if failed:
        print(
            f"WARNING: {len(failed)} CGM(s) fell back to an unrefined "
            f"block (no kME noise step, whole CGM as core) after FGM extraction "
            f"failed: {[f['CGM'] for f in failed]}"
        )
        for f in failed:
            print(f"  {f['CGM']} ({f['n_genes']} genes): {f['error']}")

    noise_df = pd.DataFrame(
        {c: NOISE_MODULE for c in ("module", "module_core", "merged_module", "merged_module_core")},
        index=noise_genes,
    )
    fgm_df = pd.concat([*fgm_dict.values(), noise_df], axis=0)
    fgm_df.attrs["failed_cgms"] = failed
    fgm_df.attrs["normalization"] = pd.DataFrame(norm_rows, columns=NORM_COLUMNS).to_dict()

    return fgm_df


def _report_fgms(fgm_df: pd.DataFrame, cgm_df: pd.DataFrame) -> None:
    print(f"\nResults:")
    print(f"  Number of fine FGMs: {len(np.unique(fgm_df.module))}")
    print(f"  Fine FGM distribution:")
    print(fgm_df.module.value_counts())
    print(f"  Number of merged FGMs: {len(np.unique(fgm_df.merged_module))}")
    print(f"  Merged FGM distribution:")
    print(fgm_df.merged_module.value_counts())

    merged_noise = fgm_df.merged_module == NOISE_MODULE
    print(
        f"  Merged-FGM noise genes: {int(merged_noise.sum())} of {len(fgm_df)} "
        f"({merged_noise.mean():.1%})"
    )
    per_cgm_noise = merged_noise.groupby(cgm_df.module.reindex(fgm_df.index).fillna(NOISE_MODULE)).agg(["sum", "mean"])
    print("  Merged-FGM noise per CGM (count, fraction):")
    print(per_cgm_noise.rename(columns={"sum": "n_noise", "mean": "fraction"}).round(3))


def _write_fgm_outputs(fgm_df: pd.DataFrame, output_dir: str, output_file_names: List[str]) -> None:
    # (column, drop noise sets, file stem, printed label); order = write order
    specs = [
        ("module", False, output_file_names[0], "Fine raw modules"),
        ("module_core", True, output_file_names[1], "Fine filtered modules"),
        ("merged_module", False, f"{output_file_names[0]}_merged", "Merged raw modules"),
        ("merged_module_core", True, f"{output_file_names[1]}_merged", "Merged filtered modules"),
    ]
    outputs = []
    for column, exclude_noise, stem, label in specs:
        d = {
            key: fgm_df[fgm_df[column] == key].index.to_list()
            for key in fgm_df[column].unique()
        }
        if exclude_noise:
            d = {k: v for k, v in d.items() if not k.endswith("-1")}
        outputs.append((d, os.path.join(output_dir, f"{stem}.gmt"), label))

    for d, path, _ in outputs:
        save_modules_to_gmt(d, path)

    print(f"\nSaved results to:")
    for _, path, label in outputs:
        print(f"  {label}: {path}")


def extract_and_save_fgms(
    expr_df: pd.DataFrame,
    cgm_df: pd.DataFrame,
    best_umap_params: Dict,
    fgm_min_clust_frac_lst: List[float],
    merge_size_multiplier_lst: List[float],
    optimize_clust_params: bool,
    gene_network: Optional[pd.DataFrame],
    output_dir: str,
    output_file_names: List[str],
    n_jobs: int = 1,
    noise_method: str = "kme",
    fgm_noise_kwargs: Optional[Dict] = None,
    diagnostics_path: Optional[str] = None,
    norm_kwargs: Optional[Dict] = None,
    return_embeddings: bool = False,
    extra_noise_genes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extract and save FGMs.

    Saves BOTH the fine (unmerged) and the merged labeling:
    `{prefix}_raw.gmt` / `{prefix}_filtered.gmt` (fine `module`/`module_core`)
    and `{prefix}_raw_merged.gmt` / `{prefix}_filtered_merged.gmt`
    (`merged_module`/`merged_module_core`, from the re-condensing merge).
    """
    _banner("FGM Extraction")

    os.makedirs(output_dir, exist_ok=True)

    fgm_df = extract_all_fgms(
        expr_df,
        cgm_df,
        best_umap_params=best_umap_params,
        optimize_clust_params=optimize_clust_params,
        fgm_min_clust_frac_lst=fgm_min_clust_frac_lst,
        merge_size_multiplier_lst=merge_size_multiplier_lst,
        gene_network=gene_network,
        n_jobs=n_jobs,
        noise_method=noise_method,
        fgm_noise_kwargs=fgm_noise_kwargs,
        norm_kwargs=norm_kwargs,
        return_embeddings=return_embeddings,
        extra_noise_genes=extra_noise_genes,
    )

    norm_path = os.path.join(
        output_dir, f"{output_file_names[0].removesuffix('_raw')}_fgm_normalization.csv"
    )
    pd.DataFrame(fgm_df.attrs["normalization"], columns=NORM_COLUMNS).to_csv(norm_path, index=False)
    print(f"  FGM normalization settings per CGM: {norm_path}")

    failed_cgms = fgm_df.attrs.get("failed_cgms", [])
    if failed_cgms:
        failed_path = os.path.join(output_dir, f"{output_file_names[0]}_failed_cgms.csv")
        pd.DataFrame(failed_cgms).to_csv(failed_path, index=False)
        print(f"  Failed CGMs (kept as unrefined blocks): {failed_path}")

    _report_fgms(fgm_df, cgm_df)

    if diagnostics_path is not None:
        diagnostics = pd.DataFrame(
            {
                "gene": fgm_df.index,
                "CGM": cgm_df.module.reindex(fgm_df.index).fillna(NOISE_MODULE).to_numpy(),
                "fine_module": fgm_df.module.to_numpy(),
                "merged_module": fgm_df.merged_module.to_numpy(),
                "kme": fgm_df.kme.to_numpy(),
                "kme_qval": fgm_df.kme_qval.to_numpy(),
            }
        )
        diagnostics.to_csv(diagnostics_path, index=False)
        print(f"  FGM kME diagnostics: {diagnostics_path}")

    _write_fgm_outputs(fgm_df, output_dir, output_file_names)

    return fgm_df


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments (Variant B: src)."""
    parser = argparse.ArgumentParser(
        description=(
            "scNestedWGCNA: hierarchical gene-module detection from single-cell "
            "RNA-seq.\n"
            "Input: normalized SEACells metacell expression (metacells x genes).\n\n"
            "How it works:\n"
            "  1. Genes are clustered on a sqrt(1-R^2) co-expression dissimilarity "
            "(DensMAP UMAP + HDBSCAN) into coarse-grained modules (CGMs).\n"
            "  2. Inside each CGM the core gene expression is regressed out, "
            "resulting in the normalized expression followed by re-clustering step into "
            "fine-grained modules (FGMs), which are merged by re-condensing the HDBSCAN tree.\n"
            "  3. Noise genes are removed (optional gene pre-filter, kME test) and "
            "the dense core of each module is kept (graph-degeneracy erosion).\n\n"
            "Parameters use fixed defaults, or are tuned with --optimize-umap/-cgm/-fgm "
            "(the CGM/FGM sweeps score by knowledge_score against a known gene regulatory "
            "network, so --gene-network-file is needed only for those).\n"
            "Output: GMT files (_raw, _filtered, _raw_merged, _filtered_merged) and "
            "diagnostics CSVs in --output-dir."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Input CSV file name containing expression data (should be in data directory)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the input data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for all produced files (GMTs and diagnostics CSVs); "
        "created if missing. Default: <project>/results/modules",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="fgm_results",
        help="Prefix for output files",
    )
    parser.add_argument(
        "--gene-network-file",
        type=str,
        default=None,
        help=(
            "Path to a gene-regulatory-network CSV (columns: target, "
            "regulator, effect; see data/grns/grn_multisim_*.csv) for "
            "knowledge_score optimization. Required only with --optimize-cgm / "
            "--optimize-fgm."
        ),
    )

    parser.add_argument(
        "--optimize-umap",
        action="store_true",
        help="Optimize UMAP parameters with Optuna. Default: off, the --umap-* values are used.",
    )
    parser.add_argument(
        "--min-clust-size-umap", type=int, default=30,
        help="HDBSCAN min_cluster_size used to score each UMAP trial (with --optimize-umap).",
    )
    parser.add_argument(
        "--min-samples-umap", type=int, default=3,
        help="HDBSCAN min_samples used to score each UMAP trial (with --optimize-umap).",
    )
    parser.add_argument(
        "--n-trials-umap", type=int, default=50,
        help="Number of Optuna trials for --optimize-umap.",
    )
    parser.add_argument(
        "--umap-n-neighbors", type=int, default=15,
        help="UMAP n_neighbors used when --optimize-umap is absent.",
    )
    parser.add_argument(
        "--umap-n-components", type=int, default=10,
        help="UMAP n_components used when --optimize-umap is absent.",
    )
    parser.add_argument(
        "--umap-min-dist", type=float, default=0.1,
        help="UMAP min_dist used when --optimize-umap is absent.",
    )

    parser.add_argument(
        "--optimize-cgm",
        action="store_true",
        help="Optimize CGM clustering parameters (knowledge_score sweep; needs "
        "--gene-network-file). Default: off, the --cgm-* values are used.",
    )
    parser.add_argument(
        "--cgm-min-clust-frac-start",
        type=float,
        default=0.03,
        help="CGM min_cluster_size search start, as a fraction of total gene count",
    )
    parser.add_argument(
        "--cgm-min-clust-frac-stop",
        type=float,
        default=0.10,
        help="CGM min_cluster_size search stop, as a fraction of total gene count",
    )
    parser.add_argument(
        "--cgm-min-clust-frac-step",
        type=float,
        default=0.02,
        help="CGM min_cluster_size search step, as a fraction of total gene count",
    )

    parser.add_argument(
        "--cgm-min-clust-size", type=int, default=50,
        help="CGM HDBSCAN min_cluster_size (gene count) used when --optimize-cgm is absent.",
    )
    parser.add_argument(
        "--cgm-min-samples", type=int, default=3,
        help="CGM HDBSCAN min_samples used when --optimize-cgm is absent.",
    )

    parser.add_argument(
        "--fgm-min-clust-frac-start",
        type=float,
        default=0.03,
        help="FGM min_cluster_size search start, as a fraction of the CGM's own gene count",
    )
    parser.add_argument(
        "--fgm-min-clust-frac-stop",
        type=float,
        default=0.20,
        help="FGM min_cluster_size search stop, as a fraction of the CGM's own gene count",
    )
    parser.add_argument(
        "--fgm-min-clust-frac-step",
        type=float,
        default=0.03,
        help="FGM min_cluster_size search step, as a fraction of the CGM's own gene count",
    )
    parser.add_argument(
        "--fgm-merge-multiplier-start",
        type=float,
        default=1.5,
        help=(
            "Start of the merge_size_multiplier search grid. The coarse "
            "'merge' cut re-condenses the same per-CGM tree at "
            "min_cluster_size * merge_size_multiplier."
        ),
    )
    parser.add_argument(
        "--fgm-merge-multiplier-stop", type=float, default=3.0,
        help="FGM merge multiplier grid: last value (merged min_cluster_size = fine size x multiplier)",
    )
    parser.add_argument(
        "--fgm-merge-multiplier-step", type=float, default=0.5,
        help="FGM merge multiplier grid: step",
    )
    parser.add_argument(
        "--optimize-fgm",
        action="store_true",
        help="Optimize FGM clustering parameters (knowledge_score sweep; needs "
        "--gene-network-file). Default: off, default FGM parameters are used.",
    )

    parser.add_argument(
        "--cgm-noise-filter",
        choices=["none", "hdbscan"],
        default="none",
        help=(
            "CGM-level noise handling. none: every gene goes to its soft-argmax "
            "CGM. hdbscan: HDBSCAN-native noise genes become module_-1."
        ),
    )
    parser.add_argument(
        "--prefilter-genes",
        action="store_true",
        help=(
            "Drop genes without a detectable co-expression partner (top-k "
            "permutation test) BEFORE UMAP/CGM clustering; CGM min_clust_size "
            "fractions are then relative to the filtered gene count. Off by default."
        ),
    )
    parser.add_argument(
        "--prefilter-k", type=int, default=5,
        help="Gene pre-filter: number of strongest correlations averaged per gene (with --prefilter-genes).",
    )
    parser.add_argument(
        "--prefilter-n-perm", type=int, default=20,
        help="Gene pre-filter: number of permutations for the null (with --prefilter-genes).",
    )
    parser.add_argument(
        "--prefilter-fdr", type=float, default=0.1,
        help="Gene pre-filter: genes with a BH q-value above this are removed (with --prefilter-genes).",
    )
    parser.add_argument(
        "--fgm-noise-method",
        choices=["hdbscan", "kme"],
        default="kme",
        help=(
            "FGM-stage noise method (the `noise_method` of extract_all_fgms). "
            "kme: soft assignments for the fine and merged FGMs, then kME of "
            "raw expression against each gene's own fine-FGM eigengene (exact "
            "null + floor) on the MERGED labels only. hdbscan: hard HDBSCAN "
            "labels for the fine and merged FGMs, native -1 = module_-1, no kME."
        ),
    )
    parser.add_argument(
        "--kme-fdr", type=float, default=0.05,
        help="kME noise test: BH FDR level (with --fgm-noise-method kme)",
    )
    parser.add_argument(
        "--kme-min", type=float, default=0.3,
        help="kME noise test: minimum kME a gene needs to stay in its module (with --fgm-noise-method kme)",
    )

    parser.add_argument(
        "--norm-k",
        type=float,
        default=2.0,
        help="Exponent k of the main dominant-signal-removal weight w = |r|^k (0 = full "
        "regression removal; larger k removes less). Default 2.",
    )
    parser.add_argument(
        "--tune-norm-k",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Choose k per CGM from --norm-k-grid by knowledge_score (needs "
        "--optimize-fgm; the FGM stage is up to ~5x slower with the default grid).",
    )
    parser.add_argument(
        "--norm-k-grid",
        type=float,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Candidate k values when --tune-norm-k is set.",
    )
    parser.add_argument(
        "--norm-corr-thr-grid",
        type=float,
        nargs="+",
        default=[0.95, 0.9, 0.85, 0.8, 0.75],
        help="Candidate INGS correlation thresholds (Tier 1 of the normalization).",
    )
    parser.add_argument(
        "--input-is-log",
        action="store_true",
        help="The input matrix is already log-transformed (e.g. normalization "
        "run with --log1p); skip the log1p step.",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for CGM-level FGM extraction. -1 for all cores, 1 for sequential (default).",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments: the normalization settings, and that the
    knowledge_score optimizations have a gene network."""
    if args.norm_k < 0 or any(k < 0 for k in args.norm_k_grid):
        raise ValueError("--norm-k and --norm-k-grid values must be >= 0")
    if not args.norm_k_grid or not args.norm_corr_thr_grid:
        raise ValueError("--norm-k-grid and --norm-corr-thr-grid must not be empty")
    if args.tune_norm_k and not args.optimize_fgm:
        raise ValueError("--tune-norm-k requires --optimize-fgm")
    if (args.optimize_cgm or args.optimize_fgm) and not args.gene_network_file:
        raise ValueError(
            "--gene-network-file is required when --optimize-cgm / --optimize-fgm is "
            "used: knowledge_score optimization needs a gene network"
        )


def _print_run_config(args: argparse.Namespace, cgm_min_clust_frac_lst: List[float], fgm_min_clust_frac_lst: List[float], merge_size_multiplier_lst: List[float]) -> None:
    _banner("scNestedWGCNA Variant B - Single-Hierarchy CGM/FGM Extraction", 60)
    print(f"\nInput file: {args.input_file}")
    print(f"Output prefix: {args.output_prefix}")
    print(f"Gene network: {args.gene_network_file or 'not used'}")
    print(f"UMAP optimization: {'ON' if args.optimize_umap else 'OFF'}")
    print(f"CGM optimization: {'ON' if args.optimize_cgm else 'OFF'}")
    print(f"FGM optimization: {'ON' if args.optimize_fgm else 'OFF'}")
    print(f"CGM min_clust_size fractions: {cgm_min_clust_frac_lst}")
    print(f"FGM min_clust_size fractions: {fgm_min_clust_frac_lst}")
    print(f"FGM merge_size multipliers: {merge_size_multiplier_lst}")
    print(f"CGM noise filter: {args.cgm_noise_filter}")
    if args.prefilter_genes:
        print(
            f"Gene pre-filter ON:  k={args.prefilter_k}, n_perm={args.prefilter_n_perm}, "
            f"fdr={args.prefilter_fdr} (CGM min_clust_size fractions are relative "
            "to the FILTERED gene count)"
        )
    print(f"FGM noise method: {args.fgm_noise_method}")
    if args.fgm_noise_method == "kme":
        print(f"  kME: fdr={args.kme_fdr}, min_kme={args.kme_min}")
    print(
        f"Normalization: confounder regression, k={args.norm_k}"
        + (f", tuning k over {args.norm_k_grid}" if args.tune_norm_k else "")
        + f", corr_thr grid {args.norm_corr_thr_grid}"
        + (", input already log" if args.input_is_log else "")
    )


def _prefilter_genes(expr_df: pd.DataFrame, args: argparse.Namespace, output_dir: str) -> pd.DataFrame:
    if not args.prefilter_genes:
        return expr_df
    print("Pre-filtering genes without co-expression partners (top-k permutation)...")
    prefilter = topk_coexpression_filter(
        expr_df,
        k=args.prefilter_k,
        n_perm=args.prefilter_n_perm,
        fdr=args.prefilter_fdr,
    )
    prefilter.to_csv(
        os.path.join(output_dir, f"{args.output_prefix}_cgm_prefilter.csv")
    )
    n_removed = int(prefilter.is_noise.sum())
    print(
        f"Removed {n_removed} of {len(prefilter)} genes "
        f"({n_removed / len(prefilter):.1%}); {len(prefilter) - n_removed} remain."
    )
    return expr_df.loc[:, ~prefilter.is_noise.to_numpy()]


def _provided_umap_params(args: argparse.Namespace) -> Optional[Dict]:
    if args.optimize_umap:
        return None
    return {
        "n_neighbors": args.umap_n_neighbors,
        "n_components": args.umap_n_components,
        "min_dist": args.umap_min_dist,
    }


def _provided_cgm_params(args: argparse.Namespace, umap_params: Dict) -> Optional[Dict]:
    """CGM params from the CLI; the embedding always uses the UMAP params."""
    if args.optimize_cgm:
        return None
    return {
        "min_clust_size": args.cgm_min_clust_size,
        "min_samples": args.cgm_min_samples,
        "n_neighbors": umap_params["n_neighbors"],
        "n_components": umap_params["n_components"],
        "min_dist": umap_params["min_dist"],
    }


def main() -> None:
    args = parse_arguments()
    validate_args(args)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    input_path = os.path.join(args.data_dir, args.input_file)
    output_files = [f"{args.output_prefix}_raw", f"{args.output_prefix}_filtered"]

    gene_network = load_gene_network(args.gene_network_file)

    cgm_min_clust_frac_lst = frac_range(
        args.cgm_min_clust_frac_start,
        args.cgm_min_clust_frac_stop,
        args.cgm_min_clust_frac_step,
    )
    fgm_min_clust_frac_lst = frac_range(
        args.fgm_min_clust_frac_start,
        args.fgm_min_clust_frac_stop,
        args.fgm_min_clust_frac_step,
    )
    merge_size_multiplier_lst = frac_range(
        args.fgm_merge_multiplier_start,
        args.fgm_merge_multiplier_stop,
        args.fgm_merge_multiplier_step,
    )

    _print_run_config(args, cgm_min_clust_frac_lst, fgm_min_clust_frac_lst, merge_size_multiplier_lst)

    try:
        print("\n" + "-" * 40)
        print("Loading and filtering data...")
        expr_df = load_and_filter_data(input_path)
        print(f"Data shape: {expr_df.shape}")

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        all_genes = expr_df.columns
        expr_df = _prefilter_genes(expr_df, args, output_dir)

        best_umap_params = get_umap_params(
            expr_df,
            args.optimize_umap,
            args.min_clust_size_umap,
            args.min_samples_umap,
            args.n_trials_umap,
            _provided_umap_params(args),
        )

        best_cgm_params = get_cgm_params(
            expr_df,
            best_umap_params,
            args.optimize_cgm,
            cgm_min_clust_frac_lst,
            gene_network,
            _provided_cgm_params(args, best_umap_params),
        )

        cgm_final_df = run_cgm_clustering(
            expr_df, best_cgm_params, cgm_noise_filter=args.cgm_noise_filter
        )

        fgm_df = extract_and_save_fgms(
            expr_df,
            cgm_final_df,
            best_umap_params,
            fgm_min_clust_frac_lst,
            merge_size_multiplier_lst,
            args.optimize_fgm,
            gene_network,
            output_dir,
            output_files,
            n_jobs=args.n_jobs,
            noise_method=args.fgm_noise_method,
            fgm_noise_kwargs={
                "kme_fdr": args.kme_fdr,
                "kme_min": args.kme_min,
            },
            diagnostics_path=os.path.join(output_dir, f"{args.output_prefix}_fgm_kme.csv"),
            extra_noise_genes=all_genes.difference(expr_df.columns).to_list(),
            norm_kwargs={
                "norm_k": args.norm_k,
                "tune_norm_k": args.tune_norm_k,
                "norm_k_lst": args.norm_k_grid,
                "norm_corr_thr_lst": args.norm_corr_thr_grid,
                "input_is_log": args.input_is_log,
            },
        )

        _banner("Workflow completed successfully!", 60)

    except Exception as e:
        print(f"\nERROR: Workflow failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    with suppress_warnings([UserWarning, FutureWarning]):
        main()
