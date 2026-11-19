import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from typing import List, Optional

# Columns whose range is below this count as constant
_CONST_TOL = 1e-12


def _to_log_array(df: pd.DataFrame, input_is_log: bool) -> np.ndarray:
    """Step 0: df -> float array on the log scale. Never mutates df."""
    X = df.to_numpy(dtype=float)
    if not input_is_log:
        if (X < 0).any():
            raise ValueError("input contains negative values; expected linear-scale counts")
        X = np.log1p(X)
    return X


def _ings_correlations(df: pd.DataFrame, core: Optional[List[str]] = None, method: str = "spearman", input_is_log: bool = False) -> Optional[pd.Series]:
    """Step 1 without the threshold: Spearman/Pearson of each gene with the PC1 eigengene.
    Returns a Series indexed by gene, or None if fewer than 2 genes."""
    Y = pd.DataFrame(_to_log_array(df, input_is_log), columns=df.columns)
    
    if core is not None:
        Y = Y.loc[:, core]
    if Y.shape[1] < 2:
        return None

    scaled = MinMaxScaler().fit_transform(RobustScaler().fit_transform(Y))
    eigengene = PCA(n_components=1).fit_transform(scaled)[:, 0]

    # PCA sign is arbitrary: orient towards the mean of the scaled genes
    if np.corrcoef(eigengene, scaled.mean(axis=1))[0, 1] < 0:
        eigengene = -eigengene
    return Y.corrwith(pd.Series(eigengene), axis=0, method=method)


def _ings_from_correlations(cor: Optional[pd.Series], corr_thr: float) -> List[str]:
    """Sorted INGS names (cor > corr_thr), or [] if fewer than 2."""
    if cor is None:
        return []
    ings = sorted(cor.index[cor > corr_thr])
    return ings if len(ings) >= 2 else []


def select_ings(df: pd.DataFrame, core: Optional[List[str]] = None, corr_thr: float = 0.9, method: str = "spearman", input_is_log: bool = False) -> List[str]:
    """Steps 0-1. Returns sorted INGS gene names, or [] if fewer than 2."""
    cor = _ings_correlations(df, core, method, input_is_log)
    return _ings_from_correlations(cor, corr_thr)


def regress_out_confounder(df: pd.DataFrame, ings: List[str], k: float = 2.0, input_is_log: bool = False) -> pd.DataFrame:
    """Steps 0, 2-4. Returns the corrected log-scale matrix with df's index (duplicates allowed) and columns."""
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    ings = list(ings)
    unknown = [g for g in ings if g not in df.columns]
    if unknown or len(set(ings)) < 2:
        raise ValueError(
            "ings must be >= 2 distinct genes of df.columns"
            + (f"; unknown: {unknown[:5]}" if unknown else "")
        )

    Y = _to_log_array(df, input_is_log)
    idx = df.columns.get_indexer(ings)
    p = len(idx)

    f = Y[:, idx].mean(axis=1)
    F = np.repeat(f[:, None], Y.shape[1], axis=1)
    F[:, idx] = (p * f[:, None] - Y[:, idx]) / (p - 1)  # leave-one-out for INGS genes

    Fc = F - F.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    sff = (Fc**2).sum(axis=0)
    syy = (Yc**2).sum(axis=0)
    sfy = (Fc * Yc).sum(axis=0)

    ok = (np.ptp(F, axis=0) > _CONST_TOL) & (np.ptp(Y, axis=0) > _CONST_TOL)
    b = np.zeros_like(sff)
    r = np.zeros_like(sff)
    b[ok] = sfy[ok] / sff[ok]
    r[ok] = np.clip(sfy[ok] / np.sqrt(sff[ok] * syy[ok]), -1.0, 1.0)
    w = np.abs(r) ** k

    return pd.DataFrame(Y - (w * b) * Fc, index=df.index, columns=df.columns)
