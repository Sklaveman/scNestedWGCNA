import warnings
import argparse

warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated", category=UserWarning
)

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from contextlib import contextmanager
import SEACells
from typing import Iterable, Iterator, List, Optional, Tuple, Type

from src.normalization import normalize_seacells_adata, write_normalization_outputs


def str2bool(value: str) -> bool:
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got '{value}'")


@contextmanager
def suppress_warnings(categories: Iterable[Type[Warning]]) -> Iterator[None]:
    with warnings.catch_warnings():
        for category in categories:
            warnings.filterwarnings("ignore", category=category)
        yield


def build_group_seacell_adata(
    adata: ad.AnnData,
    celltype_column: str = "celltype",          # Column in adata.obs containing cell type labels
    batch_column: Optional[str] = None,                   # Optional: column in adata.obs containing batch labels
    build_kernel_on: str = "X_pca",
    min_cells_per_metacell: int = 50,           # Minimum cells per metacell
    target_metacells_per_celltype: Optional[int] = None,  # Optional: Target metacells per cell type
    n_waypoint_eigs: int = 10,
    use_gpu: bool = True,
    summarize_layer: str = "raw",
    min_celltype_size: int = 10,                # Minimum cells to process a cell type
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Build SEACell metacells with cell-type-aware aggregation
    """

    # Get cell type information
    if celltype_column not in adata.obs.columns:
        raise ValueError(f"Cell type column '{celltype_column}' not found in adata.obs")

    if batch_column is not None:
        if batch_column not in adata.obs.columns:
            raise ValueError(f"Batch column '{batch_column}' not found in adata.obs")
        group_keys = (
            adata.obs[celltype_column].astype(str)
            + "__"
            + adata.obs[batch_column].astype(str)
        )
    else:
        group_keys = adata.obs[celltype_column].astype(str)

    unique_groups = group_keys.unique()
    group_counts = group_keys.value_counts()

    # Store all metacell data
    all_metacell_adata = []
    all_subset_adata = []

    # Process each group separately
    for group in unique_groups:
        if batch_column is not None:
            celltype, batch = group.split("__", 1)
        else:
            celltype, batch = group, None

        n_cells = group_counts[group]

        if n_cells < min_celltype_size:
            warnings.warn(f"Group {group} has only {n_cells} cells, skipping")
            continue

        # Calculate number of metacells for this group
        if target_metacells_per_celltype is not None:
            n_SEACells = min(target_metacells_per_celltype, n_cells // 2)
        else:
            n_SEACells = max(1, int(np.floor(n_cells / min_cells_per_metacell)))

        print(f"Processing {group}: {n_cells} cells -> {n_SEACells} metacells")

        # Subset to this group
        adata_subset = adata[group_keys == group].copy()

        # Skip if too few cells
        if adata_subset.n_obs <= n_SEACells:
            warnings.warn(
                f"Group {group} has too few cells for {n_SEACells} metacells, reducing to 1"
            )
            n_SEACells = 1

        # Set up the model for this group
        model = SEACells.core.SEACells(
            adata_subset,
            build_kernel_on=build_kernel_on,
            n_SEACells=n_SEACells,
            n_waypoint_eigs=n_waypoint_eigs,
            use_gpu=use_gpu,
            convergence_epsilon=1e-5,
        )

        model.construct_kernel_matrix()
        model.initialize_archetypes()
        model.fit()

        # Rename SEACells to include group info
        if batch is not None:
            seacell_labels = [
                f"{celltype}_{batch}_MC{i:03d}" for i in range(n_SEACells)
            ]
        else:
            seacell_labels = [f"{celltype}_MC{i:03d}" for i in range(n_SEACells)]
        adata_subset.obs["SEACell"] = adata_subset.obs["SEACell"].astype("category")
        adata_subset.obs["SEACell"] = pd.Categorical(
            [seacell_labels[i] for i in adata_subset.obs["SEACell"].cat.codes],
            categories=seacell_labels,
        )

        # Summarize metacells
        seacell_adata = SEACells.core.summarize_by_SEACell(
            adata_subset, SEACells_label="SEACell", summarize_layer=summarize_layer
        )

        # Add metadata
        seacell_adata.obs["celltype"] = celltype
        if batch is not None:
            seacell_adata.obs["batch"] = batch
        seacell_adata.obs["metacell_size"] = (
            adata_subset.obs["SEACell"]
            .value_counts()
            .reindex(seacell_adata.obs_names, fill_value=0)
            .to_numpy()
        )

        all_subset_adata.append(adata_subset)
        all_metacell_adata.append(seacell_adata)

    adata_merged = ad.concat(all_subset_adata, axis=0, join="outer")

    # Combine all metacell datasets
    if len(all_metacell_adata) == 0:
        raise ValueError("No metacells were created!")

    SEACells_adata = ad.concat(all_metacell_adata, axis=0, join="outer")

    return SEACells_adata, adata_merged


def main(
    adata_path: str,
    celltype_column: str,
    out_filename: str,
    batch_column: Optional[str] = None,
    out_adataname: Optional[str] = None,
    build_kernel_on: str = "X_pca",
    min_cells_per_metacell: int = 30,
    use_gpu: bool = False,
    summarize_layer: str = "counts",
    min_celltype_size: int = 100,
    out_norm_filename: Optional[str] = None,
    min_mean: float = 1.0,
    exclude_regex: Optional[str] = None,
    target_total: float = 1e4,
    log1p: bool = False,
) -> ad.AnnData:
    with suppress_warnings([FutureWarning, UserWarning]):
        adata = sc.read_h5ad(adata_path)
        SEACell_sym_adata, _ = build_group_seacell_adata(
            adata,
            celltype_column=celltype_column,
            batch_column=batch_column,
            build_kernel_on=build_kernel_on,
            min_cells_per_metacell=min_cells_per_metacell,
            use_gpu=use_gpu,
            summarize_layer=summarize_layer,
            min_celltype_size=min_celltype_size,
        )

        # Raw summed counts per metacell: no normalization of any kind here.
        X = SEACell_sym_adata.X
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        seacells_df = pd.DataFrame(
            X,
            index=SEACell_sym_adata.obs_names,
            columns=SEACell_sym_adata.var_names,
        )
        seacells_df.to_csv(out_filename, index=True)

        if out_adataname is not None:
            SEACell_sym_adata.write_h5ad(out_adataname)

        if out_norm_filename is not None:
            norm, sf_df, scale = normalize_seacells_adata(
                SEACell_sym_adata,
                by_batch=batch_column is not None,
                min_mean=min_mean,
                exclude_regex=exclude_regex,
                target_total=target_total,
                log1p=log1p,
            )
            write_normalization_outputs(norm, sf_df, scale, out_norm_filename)

    return SEACell_sym_adata

def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
            description="Build SEACells metacells and save their raw (unnormalized) counts."
        )
    parser.add_argument(
        "--adata-path",
        required=True,
        help="Path to adata file.",
    )
    parser.add_argument(
        "--celltype-column",
        required=True,
        help="Column name for cell type labels stored in adata.obs.",
    )
    parser.add_argument(
        "--out-filename",
        required=True,
        help=(
            "Path of the output CSV of raw metacell counts (used as given, "
            "include the .csv extension)."
        ),
    )
    parser.add_argument(
        "--out-adataname",
        default=None,
        help="Path of the .h5ad file to save the SEACells (metacell) AnnData to. Not saved if omitted.",
    )
    parser.add_argument(
        "--build-kernel-on",
        default="X_pca",
        help="Key in adata.obsm used to build the SEACells kernel (default: X_pca).",
    )
    parser.add_argument(
        "--out-norm-filename",
        default=None,
        help=(
            "Optional path of a CSV with cell-type-aware poscounts-normalized metacell "
            "counts (median metacell total = --target-total). Also writes "
            "<stem>_size_factors.csv and <stem>_group_scales.csv. The raw CSV is unaffected."
        ),
    )
    parser.add_argument("--min-mean", type=float, default=1.0, help="Genes with mean count below this are not used to estimate size factors.")
    parser.add_argument(
        "--exclude-regex",
        default=None,
        help="Case-insensitive regex of genes left out of the factor estimation (suggested: normalization.DOMINANT_GENES_REGEX).",
    )
    parser.add_argument("--target-total", type=float, default=1e4, help="Median metacell total after normalization.")
    parser.add_argument("--log1p", action="store_true", help="log1p-transform the normalized output.")
    parser.add_argument(
        "--batch-column",
        default=None,
        help="Column name for batch labels in adata.obs. If provided, metacells are built per (celltype, batch) group.",
    )
    parser.add_argument(
        "--min-cells", type=int, default=30, help="Number of min cells per metacell."
    )
    parser.add_argument(
        "--use-gpu", type=str2bool, default=False, help="Device to run SEACells on."
    )
    parser.add_argument(
        "--summarize-layer",
        default="counts",
        help="Adata layer to use to summarize SEACells.",
    )
    parser.add_argument(
        "--min-celltype-size",
        type=int,
        default=100,
        help="Min number of cells of a particular celltype to include into SEACells computation.",
    )

    args = parser.parse_args(argv)

    return args


if __name__ == "__main__":
    args = parse_arguments(argv=None)
    main(
        adata_path=args.adata_path,
        celltype_column=args.celltype_column,
        out_filename=args.out_filename,
        batch_column=args.batch_column,
        out_adataname=args.out_adataname,
        build_kernel_on=args.build_kernel_on,
        min_cells_per_metacell=args.min_cells,
        use_gpu=args.use_gpu,
        summarize_layer=args.summarize_layer,
        min_celltype_size=args.min_celltype_size,
        out_norm_filename=args.out_norm_filename,
        min_mean=args.min_mean,
        exclude_regex=args.exclude_regex,
        target_total=args.target_total,
        log1p=args.log1p,
    )
