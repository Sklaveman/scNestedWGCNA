import copy
import warnings
import numpy as np
import pandas as pd
import hdbscan
import umap
from scipy.linalg import eigh
from scipy.stats import false_discovery_control, norm, rankdata

from src.confounder_regression import regress_out_confounder

from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple, Type, Union


def get_adjacency(
    expressions: np.ndarray,
    alpha: float = 2,
    signed: bool = False,
    method: str = "pearson",
) -> np.ndarray:
    """Compute adjacency matrix from gene expression data."""
    if method.lower() == "pearson":
        # Pearson correlation
        correlations = np.corrcoef(expressions, rowvar=False)
    elif method.lower() == "spearman":
        # Spearman correlation - convert to ranks first
        # Rank each gene across samples
        ranked_expr = rankdata(expressions, axis=0)

        # Compute Pearson correlation on ranks
        correlations = np.corrcoef(ranked_expr, rowvar=False)
    else:
        raise ValueError(
            f"Method '{method}' not supported. Use 'pearson' or 'spearman'."
        )

    # Apply soft thresholding
    if signed:
        # Signed network: scale correlations from [-1, 1] to [0, 1]
        adj = ((1 + correlations) / 2) ** alpha
    else:
        # Unsigned network: use absolute correlations
        adj = np.abs(correlations) ** alpha

    return adj.astype(np.float32)


def symmetrize(square_sym_matrix: np.ndarray) -> np.ndarray:
    return (square_sym_matrix + square_sym_matrix.T) / 2


def core_decomposition(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    connectivity = data.sum(axis=0)
    indexes = np.zeros(len(connectivity), dtype=int)
    degeneracy = np.zeros(len(connectivity))
    for i in range(len(data)):
        idx = np.argmin(connectivity)
        connectivity -= data[idx]
        indexes[i] = idx
        degeneracy[i] = connectivity[idx]
        connectivity[idx] = np.inf
    return indexes, degeneracy


def errode_clusters(clusters: np.ndarray, dissim: np.ndarray) -> np.ndarray:
    new_clusters = clusters.copy()
    for i in (
        np.unique(clusters)
        if clusters.dtype.kind in ("S", "U", "O")
        else range(clusters.max() + 1)
    ):
        cluster = np.where(clusters == i)[0]
        if len(cluster) == 0:  # gap in integer labels (e.g. cluster emptied by noise calls)
            continue
        new_clusters[cluster] = "-1" if type(i) is str else -1
        if len(cluster) < 2:  # no core to extract from a singleton
            continue
        genes = dissim[cluster][:, cluster]
        indexes, degeneracy = core_decomposition(1 - genes)
        new_clusters[cluster[indexes[np.argmax(degeneracy) :]]] = i
    return new_clusters


def _rank_zscore(values: np.ndarray) -> np.ndarray:
    """Rank-transform each column (average ties), centre it and divide by its
    ddof=0 std, so Pearson on the result equals Spearman (consistent with
    `get_adjacency(method="spearman")`). Zero-variance columns become all zeros."""
    ranks = rankdata(values, axis=0)
    ranks = ranks - ranks.mean(axis=0)
    sd = ranks.std(axis=0)
    sd[sd == 0] = 1.0
    return ranks / sd


def topk_coexpression_filter(
    expr_df: pd.DataFrame,
    k: int = 5,
    n_perm: int = 20,
    n_sparsity_bins: int = 10,
    fdr: float = 0.1,
    chunk_size: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Flag genes with no detectable co-expression partner (CGM-level pre-filter).

    Statistic: mean of a gene's `k` largest Spearman correlations (signed, self
    excluded) with all other genes, computed in chunks of `chunk_size` genes so
    the G x G matrix is never held in memory (float32).

    Null: for each of `n_perm` permutations ONE full matrix is drawn with every
    gene's metacell values permuted independently (so results do not depend on
    `chunk_size`); each permuted gene is correlated with the UNPERMUTED other
    genes and gets the same top-k mean. Only an `n_perm x G` array of null
    statistics is kept. Null statistics are pooled within quantile bins of the
    gene's zero fraction; p = (1 + #pooled_null >= observed) / (1 + n_pooled);
    BH across genes; `is_noise = qval > fdr`. Zero-variance genes (constant but
    non-zero genes are not removed by `load_and_filter_data`) get p = 1 and are
    always noise.

    Caveats:
      - Pooling assumes genes of similar sparsity share a null distribution.
      - Metacells are not independent, so the p-values are somewhat
        anticonservative.
      - `k` must stay below the smallest module size you want to keep: a gene
        in a module of fewer than ~k+1 genes cannot have k real partners.
      - With few genes (G < ~300) the pooled null is small and BH can only
        reject if several genes are strongly co-expressed.
      - Genes with weak true partners (r ~ 0.3) can be dropped when G is large,
        since the null top-k mean grows with G; the removal is irreversible
        (done before UMAP / CGM clustering).

    Returns a DataFrame indexed by gene with `topk_mean, zero_frac,
    sparsity_bin, pval, qval, is_noise`.
    """
    n, G = expr_df.shape
    if G <= k:
        raise ValueError(f"need more than k={k} genes, got {G}")
    Z = _rank_zscore(expr_df.values).astype(np.float32)
    zero_var = ~Z.any(axis=0)
    rng = np.random.default_rng(random_state)

    def topk_means(query: np.ndarray) -> np.ndarray:
        out = np.empty(G, dtype=np.float32)
        for s in range(0, G, chunk_size):
            e = min(s + chunk_size, G)
            corr = query[:, s:e].T @ Z / n
            corr[np.arange(e - s), np.arange(s, e)] = -np.inf  # exclude self
            out[s:e] = np.partition(corr, G - k, axis=1)[:, G - k :].mean(axis=1)
        return out

    observed = topk_means(Z)
    null = np.empty((n_perm, G), dtype=np.float32)
    for b in range(n_perm):
        null[b] = topk_means(rng.permuted(Z, axis=0))

    zero_frac = (expr_df.values == 0).mean(axis=0)
    if np.ptp(zero_frac) == 0:
        bins = np.zeros(G, dtype=int)
    else:
        bins = pd.qcut(zero_frac, n_sparsity_bins, labels=False, duplicates="drop")
        bins = np.nan_to_num(np.asarray(bins, dtype=float), nan=0).astype(int)

    pval = np.empty(G)
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        pooled = np.sort(null[:, idx].ravel())
        count_ge = len(pooled) - np.searchsorted(pooled, observed[idx], side="left")
        pval[idx] = (1 + count_ge) / (1 + len(pooled))
    pval[zero_var] = 1.0

    qval = false_discovery_control(pval)
    return pd.DataFrame(
        {
            "topk_mean": observed,
            "zero_frac": zero_frac,
            "sparsity_bin": bins,
            "pval": pval,
            "qval": qval,
            "is_noise": (qval > fdr) | zero_var,
        },
        index=expr_df.columns,
    )


def kme_noise(
    expr_df: pd.DataFrame,
    labels: np.ndarray,
    tested: Optional[np.ndarray] = None,
    fdr: float = 0.05,
    min_kme: float = 0.3,
) -> pd.DataFrame:
    """
    Flag genes that are not related to their own module (FGM-level noise test).
    It is an alternative to the HDBSCAN's native noise (outlier) detection method.

    `expr_df` (metacells x genes) must be RAW, un-normalized expression: kME
    computed on normalized expression cannot detect noise (measured on
    ImmFocus-normalized data: decoys keep a median kME of ~0.5 there, while on
    raw expression decoys sit at ~0.07 vs ~0.87 for real genes; not re-measured
    for the confounder regression, see evaluation/decoy_noise_eval.py).
    `labels` (int per gene, `expr_df.columns` order) defines each gene's
    eigengene; `get_fgms_from_cgm` passes the FINE FGM labels, so that a gene of
    a small FGM is tested against its own FGM rather than against the PC1 of a
    merged module that a larger constituent may dominate. Label -1 means "no
    module".

    Per module (label != -1), ALL genes carrying that label build the
    eigengene: PC1 of the rank-z-scored genes, sign-aligned with their mean.
    kME is the Spearman correlation of a gene with its leave-one-out eigengene
    (recomputed without the gene, for every module size). The p-value is the
    upper tail of the exact permutation null (standardised gene vs. fixed
    standardised eigengene has mean 0 and variance 1/(n-1)):
    `norm.sf(arctanh(kme) * sqrt(n_metacells - 3))`, no permutations needed.
    BH runs over the tested genes of THIS call only, so it is per CGM when
    called once per CGM. `is_noise = (qval > fdr) | (kme < min_kme)`.

    This asks "is this gene related to its module at all"; it is not an FDR on
    module membership (genes were selected into the module because they
    correlate). `min_kme=0.3` was chosen on one benchmark (multisim 1140, where
    the floor removed 0.5% of real genes); re-check it with
    `evaluation/decoy_noise_eval.py` on other data.

    `tested` (bool per gene, default all) selects the genes that get a p-value,
    enter BH and can be flagged. Genes that are not tested, or have label -1,
    or sit in a module with fewer than 4 genes (not testable; label left as
    is) get NaN statistics and `is_noise=False`: the caller decides what to do
    with them. A tested zero-variance gene in a testable module is noise with
    NaN statistics.

    Returns a DataFrame indexed by gene with `module, kme, pval, qval, is_noise`.
    """
    n, G = expr_df.shape
    labels = np.asarray(labels)
    tested = np.ones(G, dtype=bool) if tested is None else np.asarray(tested, dtype=bool)
    Z = _rank_zscore(expr_df.values)
    zero_var = ~Z.any(axis=0)

    kme = np.full(G, np.nan)
    testable = np.zeros(G, dtype=bool)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if lab == -1 or len(idx) < 4:
            continue
        testable[idx] = True
        Zm = Z[:, idx]
        Zc = Zm - Zm.mean(axis=0)
        Zm_sum = Zm.sum(axis=1)
        gram = Zc @ Zc.T                     # n x n; leave-one-out PC1 = top eigenvector of gram - x x^T
        for pos, g in enumerate(idx):
            if not tested[g] or zero_var[g]:
                continue
            x = Zc[:, pos]
            _w, vec = eigh(gram - np.outer(x, x), subset_by_index=[n - 1, n - 1])
            e = vec[:, 0]
            rest_mean = (Zm_sum - Zm[:, pos]) / (len(idx) - 1)
            if np.dot(e, rest_mean - rest_mean.mean()) < 0:
                e = -e
            e = _rank_zscore(e[:, None])[:, 0]
            kme[g] = Zm[:, pos] @ e / n

    pval = np.full(G, np.nan)
    qval = np.full(G, np.nan)
    has_stat = ~np.isnan(kme)
    if has_stat.any():
        pval[has_stat] = norm.sf(
            np.arctanh(np.clip(kme[has_stat], -0.999999, 0.999999)) * np.sqrt(n - 3)
        )
        qval[has_stat] = false_discovery_control(pval[has_stat])

    is_noise = np.zeros(G, dtype=bool)
    is_noise[has_stat] = (qval[has_stat] > fdr) | (kme[has_stat] < min_kme)
    is_noise |= testable & tested & zero_var
    return pd.DataFrame(
        {"module": labels, "kme": kme, "pval": pval, "qval": qval, "is_noise": is_noise},
        index=expr_df.columns,
    )


def _dissimilarity(data: pd.DataFrame) -> np.ndarray:
    """sqrt(1 - signed Spearman adjacency) between the columns of `data`, float32."""
    sim = get_adjacency(data.values, alpha=2, signed=True, method="spearman")
    dissim = np.sqrt(np.round(symmetrize(1 - sim), 12)).astype(np.float32)
    del sim  # free ~N² × 4 bytes
    return dissim


def _embed(dissim: np.ndarray, n_neighbors: int, min_dist: float, n_components: int, random_state: int) -> np.ndarray:
    """DensMAP UMAP of a precomputed dissimilarity matrix."""
    return umap.UMAP(
        densmap=True,
        dens_lambda=1.0,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        low_memory=False,
        metric="precomputed",
    ).fit_transform(dissim)


def _embed_dissim(data: pd.DataFrame, n_neighbors: int, min_dist: float, n_components: int, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """Spearman dissimilarity (sqrt(1 - signed adjacency)) -> DensMAP UMAP.
    Returns `(dissim, embedding)`; shared by `get_clust`, `get_clust_hierarchy`
    and the parameter sweeps (which call it once and reuse the embedding)."""
    if n_components is None:
        n_components = np.around(np.log2(data.shape[0]), 0) + 1

    dissim = _dissimilarity(data)
    return dissim, _embed(dissim, n_neighbors, min_dist, n_components, random_state)


def get_clust(
    data: pd.DataFrame,
    min_clust_size: int = 30,
    n_neighbors: Optional[int] = None,
    min_dist: float = 0.0,
    n_components: Optional[int] = None,
    min_samples: Optional[int] = None,
    random_state: int = 42,
    hdbscan_n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray, "hdbscan.HDBSCAN"]:
    """
    Single-resolution embed-and-cluster call, used for the final fit at the
    chosen parameters (soft assignment + errode_clusters downstream), both for
    CGM extraction and for the FGM base clustering inside a CGM.

    For searching over min_cluster_size / min_samples, see
    `get_clust_hierarchy()` + `recondense_labels()` below, which build one
    HDBSCAN tree per min_samples and re-condense it per min_cluster_size.

    Returns `(dissim, embedding, clusterer)` — not the raw `clusterer.labels_`
    from this fit, since every caller immediately discards it in favor of a
    soft-reassignment pass (`hdbscan.all_points_membership_vectors`) computed
    from `clusterer` right after this call; it's already accessible via
    `clusterer.labels_` for anyone who does want the raw fit.
    """

    dissim, embedding = _embed_dissim(
        data,
        n_neighbors if n_neighbors else 2 * min_clust_size,
        min_dist,
        n_components,
        random_state,
    )

    clusterer = hdbscan.HDBSCAN(
        metric="euclidean",
        min_cluster_size=min_clust_size,
        min_samples=min_clust_size // 2 if min_samples is None else min_samples,
        prediction_data=True,
        approx_min_span_tree=False,
        gen_min_span_tree=True,
        core_dist_n_jobs=hdbscan_n_jobs,
    )
    clusterer.fit(embedding)

    return dissim, embedding, clusterer


def get_clust_hierarchy(
    data: pd.DataFrame,
    min_samples: int,
    n_neighbors: Optional[int] = None,
    min_dist: float = 0.0,
    n_components: Optional[int] = None,
    random_state: int = 42,
    hdbscan_n_jobs: int = -1,
    base_min_cluster_size: int = 5,
    embedding: Optional[np.ndarray] = None,
) -> "hdbscan.HDBSCAN":
    """
    Fit ONE HDBSCAN mutual-reachability hierarchy for a fixed `min_samples` on
    the UMAP embedding of `data`, or on `embedding` if given (then no UMAP is
    fit and the UMAP arguments are ignored; the sweeps pass the same embedding
    for every `min_samples`). `base_min_cluster_size` only seeds the initial fit's
    condensed tree (HDBSCAN's constructor requires *some* min_cluster_size) —
    it does not restrict which min_cluster_size can later be extracted via
    `recondense_labels()`, since re-condensing operates on the single-linkage
    tree, which is min_cluster_size-independent (only `min_samples` affects
    the underlying mutual-reachability graph / MST, via core distances).

    This is the primitive of the CGM/FGM parameter search: every call fits one
    HDBSCAN tree (the sweeps call it once per min_samples on a shared,
    precomputed embedding), and min_cluster_size is then swept by
    re-condensing that tree many times.

    Returns just the fitted `clusterer` — its callers (the CGM/FGM parameter
    sweeps) only ever need `clusterer.single_linkage_tree_` via
    `recondense_labels()`; the dissimilarity matrix and embedding used to get
    there aren't consumed downstream, unlike in `get_clust()`.
    """
    if embedding is None:
        _dissim, embedding = _embed_dissim(
            data,
            n_neighbors if n_neighbors else 2 * base_min_cluster_size,
            min_dist,
            n_components,
            random_state,
        )

    clusterer = hdbscan.HDBSCAN(
        metric="euclidean",
        min_cluster_size=base_min_cluster_size,
        min_samples=min_samples,
        prediction_data=False,
        approx_min_span_tree=False,
        gen_min_span_tree=True,
        core_dist_n_jobs=hdbscan_n_jobs,
    )
    clusterer.fit(embedding)

    return clusterer


def recondense_labels(
    clusterer: "hdbscan.HDBSCAN",
    min_cluster_size: int,
    cluster_selection_method: str = "eom",
    allow_single_cluster: bool = False,
    cluster_selection_epsilon: float = 0.0,
) -> np.ndarray:
    """
    Re-extract a flat, stability-optimal (excess-of-mass) HDBSCAN clustering
    at a *different* min_cluster_size from an already-fitted clusterer,
    reusing its `single_linkage_tree_` instead of recomputing the
    mutual-reachability graph / minimum spanning tree.

    This is the mechanism behind both (a) cheap min_cluster_size grid search
    (`get_clust_hierarchy()` fits the tree, this re-condenses it many times), and
    (b) the density-native FGM merge step (in `get_fgms_from_cgm()`): 
    re-condensing the same per-CGM tree at a coarser min_cluster_size
    merges fragments that are really one density-connected structure viewed
    at a lower resolution, rather than relying on eigengene correlation
    between two already-independently-extracted sub-clusters.

    Verified against the installed `hdbscan` package (see
    `hdbscan.flat._tree_to_labels`, `hdbscan.hdbscan_.HDBSCAN.dbscan_clustering`,
    and `SingleLinkageTree.get_clusters` in `hdbscan/plots.py`), which
    together confirm `single_linkage_tree_` is always populated after `.fit()`
    (independent of `min_cluster_size`) and that `condense_tree` /
    `compute_stability` / `get_clusters` from `hdbscan._hdbscan_tree` are the
    exact primitives HDBSCAN itself uses internally to go from a
    single-linkage tree to flat labels. Confirm the import path still holds
    for whatever `hdbscan` version is installed on the remote server before
    relying on this in production — it is not a fully public/versioned API.

    Returns only the labels — every caller in this codebase scores/merges by
    label alone; the per-point membership probabilities `get_clusters()` also
    produces aren't consumed anywhere here.
    """
    from hdbscan._hdbscan_tree import condense_tree, compute_stability, get_clusters

    single_linkage_tree = clusterer.single_linkage_tree_.to_numpy()
    condensed_tree = condense_tree(single_linkage_tree, min_cluster_size)
    stability_dict = compute_stability(condensed_tree)
    labels, _probabilities, _stabilities = get_clusters(
        condensed_tree,
        stability_dict,
        cluster_selection_method,
        allow_single_cluster,
        False,  # match_reference_implementation
        cluster_selection_epsilon,
        0,  # max_cluster_size (0 = unlimited)
    )
    return labels


def _soft_labels(clusterer: hdbscan.HDBSCAN) -> np.ndarray:
    """Soft-assignment label of every point: argmax of its HDBSCAN membership vector."""
    soft = hdbscan.all_points_membership_vectors(clusterer)
    return np.array([int(np.argmax(x)) for x in soft])


def recondense_soft_labels(clusterer: hdbscan.HDBSCAN, min_cluster_size: int) -> np.ndarray:
    """
    Soft-assignment version of `recondense_labels`: every point gets the argmax
    of its HDBSCAN membership vector at the re-condensed `min_cluster_size`, so
    there is no native `-1`. Works on a shallow copy of `clusterer` whose
    condensed tree and prediction data are replaced (the original is not
    modified); needs `prediction_data=True` at fit time (`get_clust` does that).
    Cluster numbering matches `recondense_labels` (sorted selected clusters).
    If no cluster is selected at this cut, all points are `-1`.
    """
    from hdbscan._hdbscan_tree import condense_tree

    c2 = copy.copy(clusterer)
    c2.condensed_tree_ = condense_tree(
        clusterer.single_linkage_tree_.to_numpy(), min_cluster_size
    )
    c2.generate_prediction_data()
    soft = hdbscan.all_points_membership_vectors(c2)
    if soft.ndim != 2:  # no cluster selected: hdbscan returns a 1-D zero array
        return np.full(len(clusterer.labels_), -1)
    return soft.argmax(axis=1)


def _union_of_fine(fine_int: np.ndarray, coarse_int: np.ndarray) -> np.ndarray:
    """
    Merge by union: every fine cluster goes, as a whole, to the coarse cluster
    that most of its genes fall into (ties: smallest label). Coarse `-1` genes
    do not vote; a fine cluster whose genes are all coarse `-1` becomes `-1`,
    and genes with fine label `-1` stay `-1`. The result therefore never has
    more clusters than `fine_int` and each merged cluster is a union of fine
    clusters (the coarse cut alone can split a fine cluster or give more
    clusters, because HDBSCAN's cluster selection is not monotonic in
    `min_cluster_size`).
    """
    fine_int, coarse_int = np.asarray(fine_int), np.asarray(coarse_int)
    merged = np.full(len(fine_int), -1, dtype=coarse_int.dtype)
    for f in np.unique(fine_int):
        if f == -1:
            continue
        idx = fine_int == f
        votes = coarse_int[idx][coarse_int[idx] != -1]
        if len(votes):
            merged[idx] = np.bincount(votes).argmax()
    return merged


NOISE_MODULE = "module_-1"  # the one noise module: CGM, kME and core-erosion noise all end here


def _prefix(name: str, labels: Iterable[int]) -> list:
    """`name_label` string for every label; label -1 (noise) is always `NOISE_MODULE`."""
    return [NOISE_MODULE if label == -1 else f"{name}_{label}" for label in labels]


def get_cgms(
    expr_df: pd.DataFrame,
    min_clust_size: int,
    min_samples: int = 3,
    n_components: Optional[int] = None,
    n_neighbors: Optional[int] = None,
    min_dist: float = 0.05,
    random_state: int = 42,
    noise_method: str = "none",
) -> Tuple[
    np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, "hdbscan.HDBSCAN"
]:
    """
    CGM extraction. `noise_method="none"`: every gene goes to its soft-argmax
    cluster (no `module_-1`). `"hdbscan"`: genes with HDBSCAN-native noise
    labels (`clusterer.labels_ == -1`) become `module_-1` and all others keep
    their native HDBSCAN label. `core_module` is the eroded dense core in both.
    Returns `(dissim, embedding, modules_df, module_counts, clusterer)`.
    """
    if noise_method not in ("none", "hdbscan"):
        raise ValueError(f"noise_method must be 'none' or 'hdbscan', got {noise_method!r}")

    print(f"Clustering with {min_clust_size = } and {min_samples = }.")
    dissim, embedding, clusterer = get_clust(
        expr_df,
        min_clust_size=min_clust_size,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        min_samples=min_samples,
        n_components=n_components,
        random_state=random_state,
    )

    if noise_method == "hdbscan":
        clusters = np.asarray(clusterer.labels_).astype(int)
    else:
        clusters = _soft_labels(clusterer)
    core_clusters = errode_clusters(clusters, dissim)

    print("Clustering done.")
    modules_df = pd.DataFrame(
        {"module": clusters, "core_module": core_clusters}, index=expr_df.columns
    )
    modules_df.module = _prefix("module", modules_df.module)
    modules_df.core_module = _prefix("module", modules_df.core_module)

    module_counts = (
        modules_df.module.value_counts()
        .to_frame()
        .reset_index()
        .sort_values("count", ascending=False)
    )
    module_counts.module = module_counts.module.astype("category")

    return dissim, embedding, modules_df, module_counts, clusterer


def _assemble_merged_labels(
    merged_int: np.ndarray,
    fine_int: np.ndarray,
    module_expr_df: pd.DataFrame,
    dissim: np.ndarray,
    parent_module_name: str,
    noise_method: str = "kme",
    kme_fdr: float = 0.05,
    kme_min: float = 0.3,
) -> pd.DataFrame:
    """
    Merged-FGM label assembly: integer merged labels -> kME noise calls with
    `noise_method="kme"` (merged labels only; `fine_int` only defines each
    gene's eigengene and is never modified) -> core erosion on INTEGER labels ->
    prefixing. With `noise_method="hdbscan"` the merged labels are used as they
    are (their native `-1` is the noise). Returns `merged_module`,
    `merged_module_core`, `kme`, `kme_qval` indexed by `module_expr_df.columns`
    (kme columns are NaN with `noise_method="hdbscan"`).
    Noise (kME calls, and non-core genes in the core column) is `NOISE_MODULE`,
    not a per-CGM label.
    """
    if noise_method not in ("hdbscan", "kme"):
        raise ValueError(f"noise_method must be 'hdbscan' or 'kme', got {noise_method!r}")

    merged = np.asarray(merged_int).copy()
    kme = np.full(len(merged), np.nan)
    kme_qval = np.full(len(merged), np.nan)
    if noise_method == "kme":
        result = kme_noise(
            module_expr_df,
            fine_int,
            tested=(merged_int != -1),
            fdr=kme_fdr,
            min_kme=kme_min,
        )
        merged[result.is_noise.to_numpy()] = -1
        kme = result.kme.to_numpy()
        kme_qval = result.qval.to_numpy()

    core = errode_clusters(merged, dissim)
    return pd.DataFrame(
        {
            "merged_module": _prefix(parent_module_name, merged),
            "merged_module_core": _prefix(parent_module_name, core),
            "kme": kme,
            "kme_qval": kme_qval,
        },
        index=module_expr_df.columns,
    )


def _select_normalization(
    module_expr_df: pd.DataFrame,
    best_umap_params: Dict,
    optimize_clust_params: bool,
    min_clust_frac_lst: Optional[List[float]],
    min_samples_lst: Optional[List[int]],
    merge_size_multiplier_lst: Optional[List[float]],
    gene_network: Optional[pd.DataFrame],
    gene_pathway_dict: Optional[Dict[str, Set[str]]],
    min_clust_size: int,
    min_samples: int,
    merge_size_multiplier: float,
    method: str,
    core: Optional[List[str]],
    hdbscan_n_jobs: int,
    norm_k: float,
    tune_norm_k: bool,
    norm_k_lst: List[float],
    norm_corr_thr_lst: Optional[List[float]],
    input_is_log: bool,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Tier 1 / Tier 2 normalization choice of `get_fgms_from_cgm()` (same arguments).
    Returns `(normalized_module, norm_settings, final_cluster_params)`;
    `final_cluster_params` has `min_clust_size`, `min_samples`,
    `merge_size_multiplier`, `n_neighbors`, `n_components`, `min_dist` for the
    final fit (UMAP values from `best_umap_params` unless Tier 2 ran).
    """
    # lazy: parameter_optimization imports this module; resolved at call time
    from src.parameter_optimization import (
        _sizes_from_fracs,
        optimize_fgm_clustering_params,
        select_confounder_params,
    )

    n_genes_in_cgm = module_expr_df.shape[1]
    final_cluster_params = None  # set by Tier 2 only

    print("Normalization parameter selection (Tier 1: INGS bootstrap stability).")
    tier1_kwargs = {} if norm_corr_thr_lst is None else {"corr_thr_lst": norm_corr_thr_lst}
    shortlist = select_confounder_params(
        module_expr_df,
        core=core,
        method=method,
        input_is_log=input_is_log,
        **tier1_kwargs,
    )

    def _normalize(ings: List[str], k: float) -> pd.DataFrame:
        return regress_out_confounder(module_expr_df, ings, k=k, input_is_log=input_is_log)

    norm_settings = {
        "corr_thr": np.nan,
        "k": np.nan,
        "n_ings": np.nan,
        "knowledge_score": np.nan,
    }

    if not shortlist:
        print(
            "WARNING: no valid INGS candidates. Falling back to raw module expression."
        )
        normalized_module = module_expr_df.copy()
    elif not optimize_clust_params:
        # No downstream clustering optimization requested: Tier 2 (which needs
        # a full clustering-parameter sweep per candidate) doesn't run. Take
        # Tier 1's best candidate by stability as-is.
        chosen = shortlist[0]
        normalized_module = _normalize(chosen["ings"], norm_k)
        norm_settings.update(corr_thr=chosen["corr_thr"], k=norm_k, n_ings=chosen["n_ings"])
        print(
            f"Normalization (no clustering optimization): corr_thr={chosen['corr_thr']}, "
            f"k={norm_k}, n_INGS={chosen['n_ings']}"
        )
    else:
        if min_clust_frac_lst is None:
            min_clust_frac_lst = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
        if min_samples_lst is None:
            min_samples_lst = list(range(1, 4))
        if merge_size_multiplier_lst is None:
            merge_size_multiplier_lst = [1.5, 2.0, 2.5, 3.0]

        min_clust_size_lst = _sizes_from_fracs(min_clust_frac_lst, n_genes_in_cgm, 2)

        # norm_k first, so ties (strict >) fall back to the default k
        k_lst = [norm_k] + [k for k in norm_k_lst if k != norm_k] if tune_norm_k else [norm_k]

        print(
            f"Submodules identification (Tier 2: scoring {len(shortlist)} "
            f"corr_thr x {len(k_lst)} k candidates by downstream knowledge_score)."
        )
        best_overall = None
        table = []
        for candidate in shortlist:
            for k in k_lst:
                # matrices are built per candidate and dropped, never all held at once
                params, score = optimize_fgm_clustering_params(
                    _normalize(candidate["ings"], k),
                    best_umap_params=best_umap_params,
                    min_clust_size_lst=min_clust_size_lst,
                    min_samples_lst=min_samples_lst,
                    merge_size_multiplier_lst=merge_size_multiplier_lst,
                    gene_network=gene_network,
                    gene_pathway_dict=gene_pathway_dict,
                    hdbscan_n_jobs=hdbscan_n_jobs,
                )
                table.append((candidate["corr_thr"], k, score))
                print(
                    f"Candidate corr_thr={candidate['corr_thr']}, k={k}: "
                    f"best knowledge_score={score:.4f}"
                )
                if best_overall is None or score > best_overall["score"]:
                    best_overall = {
                        "candidate": candidate,
                        "k": k,
                        "clust_params": params,
                        "score": score,
                    }

        if tune_norm_k:
            print(
                pd.DataFrame(table, columns=["corr_thr", "k", "knowledge_score"])
                .round(4)
                .to_string(index=False)
            )
        chosen = best_overall["candidate"]
        print(
            f"Selected normalization: corr_thr={chosen['corr_thr']}, k={best_overall['k']}, "
            f"n_INGS={chosen['n_ings']} "
            f"(overall best knowledge_score={best_overall['score']:.4f})"
        )
        norm_settings.update(
            corr_thr=chosen["corr_thr"],
            k=best_overall["k"],
            n_ings=chosen["n_ings"],
            knowledge_score=best_overall["score"],
        )

        normalized_module = _normalize(chosen["ings"], best_overall["k"])
        final_cluster_params = {
            key: best_overall["clust_params"][key]
            for key in (
                "min_clust_size", "min_samples", "merge_size_multiplier",
                "n_neighbors", "n_components", "min_dist",
            )
        }

    if final_cluster_params is None:
        final_cluster_params = {
            "min_clust_size": min_clust_size,
            "min_samples": min_samples,
            "merge_size_multiplier": merge_size_multiplier,
            "n_neighbors": best_umap_params["n_neighbors"] // 2 + 1,
            "n_components": best_umap_params["n_components"],
            "min_dist": best_umap_params["min_dist"],
        }
    return normalized_module, norm_settings, final_cluster_params


def get_fgms_from_cgm(
    module_expr_df: pd.DataFrame,
    best_umap_params: Dict,
    parent_module_name: str,
    optimize_clust_params: bool,
    min_clust_frac_lst: List[float] = None,
    min_samples_lst: List[int] = None,
    merge_size_multiplier_lst: List[float] = None,
    gene_network: Optional[pd.DataFrame] = None,
    gene_pathway_dict: Optional[Dict[str, Set[str]]] = None,
    min_clust_size: int = 5,
    min_samples: int = 3,
    merge_size_multiplier: float = 2.5,
    method: str = "spearman",
    core: Optional[List[str]] = None,
    hdbscan_n_jobs: int = -1,
    noise_method: str = "kme",
    kme_fdr: float = 0.05,
    kme_min: float = 0.3,
    return_raw_embedding: bool = True,
    norm_k: float = 2.0,
    tune_norm_k: bool = False,
    norm_k_lst: List[float] = [0, 1, 2, 3, 4],
    norm_corr_thr_lst: Optional[List[float]] = None,
    input_is_log: bool = False,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], np.ndarray]:
    """
    Variant B FGM extraction for one CGM. Returns `(submodules_df, embedding_raw,
    embedding_norm)`.

    `submodules_df` (indexed by gene): fine `module` / `module_core`,
    `merged_module` / `merged_module_core` (see Merge) and the diagnostics
    `kme` / `kme_qval` (NaN with `noise_method="hdbscan"`). `noise_method`
    (same name as in `get_cgms`) picks the noise behaviour of the whole stage:
      * `"hdbscan"`: hard HDBSCAN labels for the fine and the merged cut, their
        native `-1` is `module_-1`; no kME.
      * `"kme"` (default): soft assignments (argmax membership vector) for the
        fine and the merged cut, so neither has native noise; the kME test
        (`kme_fdr`, `kme_min`) then sets `module_-1` in the MERGED labels only.
    Labels are `{parent_module_name}_{label}`; noise is `NOISE_MODULE`. The chosen normalization is in
    `submodules_df.attrs["normalization"]` (`corr_thr`, `k`, `n_ings`,
    `knowledge_score`; NaN where not applicable, all NaN on the raw fallback).
    `embedding_norm` is the UMAP fit on the confounder-regressed expression
    that was clustered; `embedding_raw` the same fit on the unnormalized
    `module_expr_df` (None with `return_raw_embedding=False`), for `plot_3d()`.

    Normalization: log-space per-gene regression on the module's confounder
    signal (`regress_out_confounder()`, removal weight `|r|^norm_k`).
    `select_confounder_params()` (Tier 1) shortlists `corr_thr` values by
    bootstrap stability of the INGS set; no shortlist falls back to the raw
    expression. `input_is_log=True` skips the `log1p` step.

    Two regimes:
      * `optimize_clust_params=False`: the best Tier 1 candidate with
        `k = norm_k`, clustering with `min_clust_size` / `min_samples` /
        `merge_size_multiplier`, UMAP from `best_umap_params` (`n_neighbors`
        halved + 1).
      * `optimize_clust_params=True` (Tier 2): every shortlisted candidate (x
        `norm_k_lst` with `tune_norm_k=True`; `norm_k` first, ties keep the
        earlier candidate, strict `>`) is scored by the `knowledge_score` of
        `optimize_fgm_clustering_params()`, which searches `min_clust_size` as
        fractions of the CGM's gene count (`min_clust_frac_lst`),
        `min_samples_lst` and `merge_size_multiplier_lst`; the winner's
        parameters (UMAP included) are used. `tune_norm_k` without
        `optimize_clust_params` raises ValueError.

    Merge: `merged_module` re-condenses the SAME HDBSCAN tree as the fine
    labels at `max(min_clust_size + 1, round(min_clust_size *
    merge_size_multiplier))` (`recondense_labels` for `"hdbscan"`,
    `recondense_soft_labels` for `"kme"`). That coarse cut is only a vote:
    `_union_of_fine` sends each fine FGM as a whole to the coarse cluster most
    of its genes fall into, so every merged FGM is a union of fine FGMs and
    there are never more merged than fine FGMs.
    The kME noise calls (see `kme_noise()`) apply to the merged
    labels only, never to the fine columns. Noise genes are labelled
    `NOISE_MODULE` (`module_-1`), the same for every CGM; the fine `module` keeps
    their origin. The same label marks the non-core genes of `module_core` /
    `merged_module_core`.
    """

    if noise_method not in ("hdbscan", "kme"):
        raise ValueError(f"noise_method must be 'hdbscan' or 'kme', got {noise_method!r}")
    if tune_norm_k and not optimize_clust_params:
        raise ValueError(
            "tune_norm_k=True needs optimize_clust_params=True "
            "(k is scored by the clustering search)"
        )

    normalized_module, norm_settings, final = _select_normalization(
        module_expr_df, best_umap_params, optimize_clust_params,
        min_clust_frac_lst, min_samples_lst, merge_size_multiplier_lst,
        gene_network, gene_pathway_dict, min_clust_size, min_samples,
        merge_size_multiplier, method, core, hdbscan_n_jobs, norm_k,
        tune_norm_k, norm_k_lst, norm_corr_thr_lst, input_is_log,
    )
    min_clust_size, min_samples = final["min_clust_size"], final["min_samples"]
    merge_size_multiplier = final["merge_size_multiplier"]
    n_neighbors_final = final["n_neighbors"]
    n_components_final = final["n_components"]
    min_dist_final = final["min_dist"]

    # Final, proper single fit at the chosen (or default) parameters — same
    # codepath as CGM extraction, so soft-assignment / errode_clusters parity
    # with the rest of the pipeline is preserved regardless of how the
    # parameters were chosen.
    dissim, embedding_norm, clusterer = get_clust(
        normalized_module,
        min_clust_size=min_clust_size,
        min_samples=min_samples,
        n_neighbors=n_neighbors_final,
        min_dist=min_dist_final,
        n_components=n_components_final,
        random_state=42,
        hdbscan_n_jobs=hdbscan_n_jobs,
    )

    # Same fit, on the CGM's unnormalized expression instead of the
    # normalized one, purely for visual before/after comparison
    # (see plot_3d() in the docstring above); only the UMAP part is needed.
    # Skipped (None) with return_raw_embedding=False, as the CLI does.
    embedding_raw = None
    if return_raw_embedding:
        _dissim_raw, embedding_raw = _embed_dissim(
            module_expr_df, n_neighbors_final, min_dist_final, n_components_final, 42
        )

    subclusters = clusterer.labels_ if noise_method == "hdbscan" else _soft_labels(clusterer)
    core_subclusters = errode_clusters(subclusters, dissim)

    print("Clustering done.\n")

    submodules_df = pd.DataFrame(
        {"module": subclusters, "module_core": core_subclusters},
        index=normalized_module.columns,
    )
    submodules_df.module = _prefix(parent_module_name, submodules_df.module)
    submodules_df.module_core = _prefix(parent_module_name, submodules_df.module_core)

    coarse_min_cluster_size = max(
        min_clust_size + 1, round(min_clust_size * merge_size_multiplier)
    )
    recondense = recondense_labels if noise_method == "hdbscan" else recondense_soft_labels
    merged_int = _union_of_fine(
        subclusters, recondense(clusterer, min_cluster_size=coarse_min_cluster_size)
    )
    merged_df = _assemble_merged_labels(
        merged_int,
        subclusters,
        module_expr_df,
        dissim,
        parent_module_name,
        noise_method=noise_method,
        kme_fdr=kme_fdr,
        kme_min=kme_min,
    )
    submodules_df = pd.concat([submodules_df, merged_df], axis=1)
    submodules_df.attrs["normalization"] = norm_settings  # after concat, which may drop attrs

    return submodules_df, embedding_raw, embedding_norm


def save_modules_to_gmt(
    data: Union[pd.DataFrame, dict[str, list[str]]],
    output_filename: str = "gene_modules.gmt",
) -> None:
    """
    Saves gene modules from a DataFrame or a dictionary to a GMT file.

    Args:
        data: Pandas DataFrame with columns `gene` and `module` (written sorted
              by module), or a dictionary where keys are module names and values
              are lists of gene symbols (written in dict order).
        output_filename: Name of the output GMT file (with a default: "gene_modules.gmt").
    """

    if isinstance(data, pd.DataFrame):
        if not all(col in data.columns for col in ["gene", "module"]):
            raise ValueError(
                "DataFrame must contain columns named 'gene' and 'module'."
            )

        modules = data.groupby("module")["gene"].apply(list).items()

    elif isinstance(data, dict):
        modules = data.items()

    else:
        raise TypeError("Input data must be a Pandas DataFrame or a dictionary.")

    with open(output_filename, "w") as f:
        for module_name, genes in modules:
            genes_str = "\t".join(genes)
            f.write(f"{module_name}\t{module_name}\t{genes_str}\n")


def plot_3d(X: np.ndarray, y: np.ndarray, title: str = "Gene embedding visualization", dot_size: int = 5) -> None:
    """
    Interactive 3D scatter of genes colored by module assignment, using the
    first 3 columns of an embedding (e.g. the first 3 UMAP components from
    `get_clust()`/`get_fgms_from_cgm()`).

    For comparing unnormalized vs. confounder-regressed structure within a
    CGM, call this twice on `get_fgms_from_cgm()`'s two embedding outputs
    (colored by the same `submodules_df["module"]` labels so the two plots
    are directly comparable):

        submodules_df, embedding_raw, embedding_norm = get_fgms_from_cgm(...)
        plot_3d(embedding_raw, submodules_df["module"].values, title="Unnormalized")
        plot_3d(embedding_norm, submodules_df["module"].values, title="Regression-normalized")
    """
    from plotly.offline import iplot
    import plotly.graph_objs as go
    from plotly.colors import qualitative

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    digits_3d = pd.DataFrame(
        {
            "x": X.iloc[:, 0],
            "y": X.iloc[:, 1],
            "z": X.iloc[:, 2],
            "label": y,
            "index_name": X.index.astype(str),
        }
    )

    if y.dtype.kind in ("U", "S"):
        unique_labels = sorted(np.unique(y), key=lambda x: int(x.split("_")[-1]))
    else:
        unique_labels = np.unique(y)

    n_classes = len(unique_labels)

    if n_classes <= 10:
        colors = qualitative.Plotly
    elif n_classes <= 24:
        colors = qualitative.Dark24
    else:
        colors = qualitative.Alphabet * (n_classes // 26 + 1)

    data = []
    for i, label_name in enumerate(unique_labels):
        subset = digits_3d[digits_3d["label"] == label_name]
        color = colors[i]

        trace = go.Scatter3d(
            x=subset["x"],
            y=subset["y"],
            z=subset["z"],
            mode="markers",
            marker={"size": dot_size, "color": color},
            name=str(label_name),
            text=subset["index_name"],
            hovertemplate=(
                "<b>Gene:</b> %{text}<br>"
                + "<b>X:</b> %{x}<br>"
                + "<b>Y:</b> %{y}<br>"
                + "<b>Z:</b> %{z}<br>"
                + "<extra></extra>"
            ),
        )
        data.append(trace)

    layout = go.Layout(title=title, width=900, height=600)
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, show_link=False)


@contextmanager
def suppress_warnings(categories: Iterable[Type[Warning]]) -> Iterator[None]:
    with warnings.catch_warnings():
        for category in categories:
            warnings.filterwarnings("ignore", category=category)
        yield
