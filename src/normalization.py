import os
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:
    import anndata as ad

DOMINANT_GENES_REGEX = r"^(MT-|RP[SL]\d|IG[HKL][VDJC]|HB[AB])"

MIN_GENES = 50


def poscounts_size_factors(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """X: samples x genes, non-negative. Returns size factors with geometric mean 1."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    pos = X > 0
    logX = np.log(np.where(pos, X, 1.0))
    log_ref = logX.sum(axis=0) / n  # zeros count in the denominator
    ratios = np.where(pos, logX - log_ref, np.nan)  # zeros don't vote
    keep = pos.any(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", RuntimeWarning
        )  # all-NaN row -> NaN, caught by callers
        sf = np.exp(np.nanmedian(ratios[:, keep], axis=1))
    return sf / np.exp(np.log(sf).mean())


def _check_finite_positive(sf: Union[np.ndarray, pd.Series], what: str) -> None:
    sf = np.asarray(sf, dtype=float)
    if not (np.isfinite(sf).all() and (sf > 0).all()):
        raise ValueError(f"{what}: size factors are not all finite and > 0")


def _validate_counts(counts: pd.DataFrame) -> None:
    if counts.index.has_duplicates:
        raise ValueError("counts has duplicate metacell names")
    
    values = counts.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("counts contains NaN")
    
    if (values < 0).any():
        raise ValueError("counts contains negative values")
    
    totals = values.sum(axis=1)
    if (totals == 0).any():
        zero = list(counts.index[totals == 0])
        raise ValueError(f"metacell(s) with zero total count: {zero}")
    
    if not np.allclose(values, np.round(values)):
        warnings.warn("counts are not integers; expected summed raw UMI counts")


def celltype_aware_size_factors(counts: pd.DataFrame, groups: pd.Series, min_mean: float = 1.0, exclude_regex: Optional[str] = None) -> Tuple[pd.Series, pd.Series]:
    """counts: metacells x genes DataFrame (summed raw UMIs). groups: Series indexed like counts.

    Returns (size_factors Series with geometric mean 1, group_scale Series).
    Factors are estimated on genes with mean count >= `min_mean` (and not matching
    `exclude_regex`, case-insensitive) but are meant to be applied to all genes."""
    if set(groups.index) != set(counts.index) or groups.index.has_duplicates:
        raise ValueError(
            "counts and groups must have the same metacells (unique index)"
        )
    groups = groups.loc[counts.index]
    _validate_counts(counts)

    C = counts
    if exclude_regex is not None:
        drop = C.columns.str.match(exclude_regex, case=False)
        print(
            f"exclude_regex removes {int(drop.sum())} genes from the estimation"
            + (f" (e.g. {', '.join(C.columns[drop][:5])})" if drop.any() else "")
        )
        C = C.loc[:, ~drop]
    C = C.loc[:, C.mean(axis=0) >= min_mean]
    if C.shape[1] < MIN_GENES:
        raise ValueError(
            f"only {C.shape[1]} genes left for estimation after min_mean/exclude_regex "
            f"filtering (need >= {MIN_GENES})"
        )

    within = pd.Series(np.nan, index=C.index)
    profiles = {}
    for g in groups.unique():
        idx = groups.index[groups == g]
        Cg = C.loc[idx]
        s = poscounts_size_factors(Cg.values) if len(idx) > 1 else np.ones(1)
        _check_finite_positive(s, f"within-group factors of group '{g}'")
        within.loc[idx] = s
        profiles[g] = Cg.div(s, axis=0).mean(axis=0)
    P = pd.DataFrame(profiles).T
    P = P.loc[:, (P > 0).all(axis=0)]
    if P.shape[1] < MIN_GENES:
        warnings.warn(
            f"only {P.shape[1]} genes are positive in every group profile "
            f"(< {MIN_GENES}); group scales are unreliable"
        )
    scale = pd.Series(poscounts_size_factors(P.values), index=P.index, name="scale")
    _check_finite_positive(scale, "group scales")
    sf = within * groups.map(scale)
    sf = sf / np.exp(np.log(sf).mean())
    _check_finite_positive(sf, "final")
    return sf, scale


def normalize_counts(counts: pd.DataFrame, size_factors: pd.Series, target_total: float = 1e4, log1p: bool = False) -> pd.DataFrame:
    """counts / size_factors, times one constant so the median metacell total is
    `target_total`; optionally log1p of that. All genes are kept."""
    if set(size_factors.index) != set(counts.index):
        raise ValueError("counts and size_factors must have the same metacells")
    sf = size_factors.loc[counts.index]
    _check_finite_positive(sf, "normalize_counts")
    norm = counts.div(sf, axis=0)
    norm = norm * (target_total / np.median(norm.sum(axis=1)))
    return np.log1p(norm) if log1p else norm


def groups_from_metacell_names(index: pd.Index) -> pd.Series:
    """Group = metacell name before the last '_' (pop2_MC012 -> pop2)."""
    names = pd.Index(index).astype(str)
    if not names.str.contains("_").all():
        bad = list(names[~names.str.contains("_")][:5])
        raise ValueError(f"metacell names without '_', cannot derive groups: {bad}")
    return pd.Series([n.rsplit("_", 1)[0] for n in names], index=names, name="group")


def normalize_metacell_counts(
    counts: pd.DataFrame, groups: pd.Series, min_mean: float = 1.0, exclude_regex: Optional[str] = None, target_total: float = 1e4, log1p: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """celltype_aware_size_factors + normalize_counts.
    Returns (normalized DataFrame, size_factors DataFrame[metacell, group, size_factor], group_scale).
    """
    sf, scale = celltype_aware_size_factors(counts, groups, min_mean, exclude_regex)
    norm = normalize_counts(counts, sf, target_total, log1p)
    sf_df = pd.DataFrame(
        {
            "metacell": counts.index,
            "group": groups.loc[counts.index].to_numpy(),
            "size_factor": sf.loc[counts.index].to_numpy(),
        }
    )
    return norm, sf_df, scale


def normalize_seacells_adata(
    adata_mc: "ad.AnnData",
    by_batch: bool = False,
    min_mean: float = 1.0,
    exclude_regex: Optional[str] = None,
    target_total: float = 1e4,
    log1p: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Normalize a SEACells metacell AnnData (summed counts in `layers["raw"]`, else `.X`).
    Groups: obs["celltype"], or obs["celltype"] + "__" + obs["batch"] when `by_batch`
    (the same labels `build_group_seacell_adata` groups by)."""
    X = adata_mc.layers["raw"] if "raw" in adata_mc.layers else adata_mc.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    counts = pd.DataFrame(X, index=adata_mc.obs_names, columns=adata_mc.var_names)
    groups = adata_mc.obs["celltype"].astype(str)
    if by_batch:
        groups = groups + "__" + adata_mc.obs["batch"].astype(str)
    groups = pd.Series(groups.to_numpy(), index=adata_mc.obs_names, name="group")
    return normalize_metacell_counts(
        counts, groups, min_mean, exclude_regex, target_total, log1p
    )


def write_normalization_outputs(norm: pd.DataFrame, size_factors_df: pd.DataFrame, group_scale: pd.Series, out_path: str) -> None:
    """Write the normalized CSV plus `<stem>_size_factors.csv` and `<stem>_group_scales.csv`."""
    stem = os.path.splitext(out_path)[0]
    norm.to_csv(out_path, index=True)
    size_factors_df.to_csv(f"{stem}_size_factors.csv", index=False)
    group_scale.rename_axis("group").rename("scale").reset_index().to_csv(
        f"{stem}_group_scales.csv", index=False
    )
