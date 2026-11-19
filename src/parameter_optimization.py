import numpy as np
import pandas as pd
import networkx as nx
from statistics import harmonic_mean
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


from src.utils import (
    get_adjacency,
    symmetrize,
    get_cgms,
    get_clust_hierarchy,
    _embed_dissim,
    recondense_labels,
    _union_of_fine,
)
from src.confounder_regression import _ings_correlations, _ings_from_correlations
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import optuna

# (ei, ej, n_edges, n_genes) from precompute_knowledge_edges()
_KnowledgeCache = Tuple[np.ndarray, np.ndarray, int, int]


def density_preservation_correlation(X: np.ndarray, X_embedded: np.ndarray, n_neighbors: int = 15, metric: str = "euclidean") -> float:
    """
    Computes the density preservation correlation metric for an embedding.
    """

    n_samples = X.shape[0]
    if n_samples != X_embedded.shape[0]:
        raise ValueError("X and X_embedded must have the same number of samples.")
    if n_neighbors >= n_samples:
        raise ValueError("n_neighbors must be less than the number of samples.")

    nn_orig = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric).fit(X)
    distances_orig, _ = nn_orig.kneighbors(X)
    r_orig = distances_orig[:, n_neighbors]

    nn_emb = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric).fit(
        X_embedded
    )
    distances_emb, _ = nn_emb.kneighbors(X_embedded)
    r_embedded = distances_emb[:, n_neighbors]

    epsilon = 1e-12
    log_r_orig = np.log(r_orig + epsilon)
    log_r_embedded = np.log(r_embedded + epsilon)

    corr = np.corrcoef(log_r_orig, log_r_embedded)[0, 1]

    return corr


def build_knowledge_graph(gene_list: List[str], gene_pathway_dict: Dict[str, Set[str]]) -> nx.Graph:

    G = nx.Graph()
    G.add_nodes_from(list(gene_pathway_dict.keys()))

    for gene_i, gene_v in combinations(gene_list, 2):
        shared_pathways = gene_pathway_dict[gene_i] & gene_pathway_dict[gene_v]
        if shared_pathways:
            G.add_edge(
                gene_i,
                gene_v,
                shared_pathways=list(shared_pathways),
                num_shared=len(shared_pathways),
            )

    return G


def precompute_knowledge_edges(expr_df: pd.DataFrame, gene_network: pd.DataFrame) -> Optional[_KnowledgeCache]:
    """Edge index arrays for `compute_knowledge_score(..., knowledge_cache=...)`.

    Same edge filtering/dedup as the `gene_network` path there. Returns
    `(ei, ej, n_edges, n_genes)`: deduplicated non-self edges as column positions
    (ei < ej), the edge-set size INCLUDING self-pairs (a, a), and the gene count.
    Returns None if the columns are not unique (use the uncached path).
    """
    cols = expr_df.columns
    if not cols.is_unique:
        return None
    pos = {g: i for i, g in enumerate(cols)}
    edges = {
        (min(pos[a], pos[b]), max(pos[a], pos[b]))
        for a, b in zip(gene_network.iloc[:, 0], gene_network.iloc[:, 1])
        if a in pos and b in pos
    }
    n_edges = len(edges)
    edges = [e for e in edges if e[0] != e[1]]
    arr = np.array(edges, dtype=np.int64).reshape(-1, 2)
    return arr[:, 0], arr[:, 1], n_edges, len(cols)


def _knowledge_score_cached(clusters: np.ndarray, cache: _KnowledgeCache) -> float:
    ei, ej, n_edges, n_genes = cache
    uniq, inv = np.unique(clusters, return_inverse=True)
    if clusters.dtype.type is np.str_:
        noise = np.array(["-1" in c for c in uniq], dtype=bool)
    else:
        noise = np.array([c == -1 for c in uniq], dtype=bool)
    sizes = np.bincount(inv, minlength=len(uniq))[~noise].tolist()
    n_coexp = sum(n * (n - 1) // 2 for n in sizes)
    cp_kp = int(((inv[ei] == inv[ej]) & ~noise[inv[ei]]).sum())

    cp_nkp = n_coexp - cp_kp
    ncp_kp = n_edges - cp_kp
    ncp_nkp = n_genes * (n_genes - 1) // 2 - n_coexp - ncp_kp
    if cp_nkp == 0 or ncp_kp == 0:
        return 0.0
    return (cp_kp * ncp_nkp) / (cp_nkp * ncp_kp)


def compute_knowledge_score(
    expr_df: pd.DataFrame, clusters: np.ndarray, gene_network: Optional[pd.DataFrame] = None, gene_pathway_dict: Optional[Dict[str, Set[str]]] = None, knowledge_cache: Optional[_KnowledgeCache] = None
) -> float:
    """Odds ratio of co-expressed vs. known gene pairs (0.0 if undefined).

    `knowledge_cache` (optional, from `precompute_knowledge_edges(expr_df,
    gene_network)`) speeds up repeated calls on the same genes; it gives the
    identical score and is only used on the `gene_network` path.
    """
    if (
        knowledge_cache is not None
        and gene_network is not None
        and gene_pathway_dict is None
        and len(clusters) == knowledge_cache[3]
    ):
        return _knowledge_score_cached(clusters, knowledge_cache)

    gene_lst = expr_df.columns.to_list()

    if gene_pathway_dict is not None and gene_network is None:
        G_knowledge = build_knowledge_graph(
            gene_lst, gene_pathway_dict=gene_pathway_dict
        )
        knowledge_edges = set(G_knowledge.edges())
    elif gene_pathway_dict is None and gene_network is not None:
        gene_set = set(gene_lst)
        # first two columns by position (target, regulator)
        knowledge_edges = {
            tuple(sorted(pair))
            for pair in zip(gene_network.iloc[:, 0], gene_network.iloc[:, 1])
            if pair[0] in gene_set and pair[1] in gene_set
        }
    else:
        raise ValueError("Either gene network or gene pathways must be provided.")

    coexp_pairs = set()
    for cluster_id in np.unique(clusters):
        if clusters.dtype.type is np.str_:
            if "-1" in cluster_id:
                continue
        else:
            if cluster_id == -1:
                continue

        cluster_indices = np.where(clusters == cluster_id)[0]
        if len(cluster_indices) < 2:
            continue

        cluster_genes = sorted(expr_df.columns[cluster_indices])
        coexp_pairs.update(combinations(cluster_genes, 2))

    cp_kp = len(coexp_pairs & knowledge_edges)
    cp_nkp = len(coexp_pairs - knowledge_edges)

    total_pairs = len(gene_lst) * (len(gene_lst) - 1) // 2
    ncp_kp = len(knowledge_edges) - cp_kp
    ncp_nkp = total_pairs - len(coexp_pairs) - ncp_kp

    if cp_nkp == 0 or ncp_kp == 0:
        return 0.0

    knowledge_score = (cp_kp * ncp_nkp) / (cp_nkp * ncp_kp)

    return knowledge_score


def optimize_umap_params(
    expr_df: pd.DataFrame,
    min_clust_size: int = 30,
    min_samples: int = 3,
    n_trials: int = 30,
    n_startup_trials: int = 4,
    n_warmup_steps: int = 6,
    reduction_factor: int = 2,
) -> dict:
    """
    Optuna search over the embedding parameters (n_neighbors/n_components/
    min_dist); each combination needs its own UMAP fit, so unlike
    min_cluster_size/min_samples it is not swept by re-condensing.
    """

    def objective(trial: optuna.Trial, expr_df: pd.DataFrame) -> float:
        max_components = np.around(np.log2(expr_df.shape[0]), 0) + 1
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 10, 20),
            "n_components": trial.suggest_int("n_components", 4, int(max_components)),
            "min_dist": trial.suggest_float("min_dist", 0.0, 0.1),
            "min_clust_size": min_clust_size,
            "min_samples": min_samples,
        }

        metrics = {
            "dbcv": [],
            "trustworthiness": [],
            "density_preservation": [],
            "f1_consistency": [],
        }

        module_labels = []
        for i in range(5):
            dissim, embedding, modules, _module_counts, clusterer = get_cgms(
                expr_df,
                random_state=trial.number * 5 + i,
                **params,
            )

            embedded_sim = get_adjacency(embedding.T, alpha=2, signed=True)
            embedded_dissim = np.sqrt(np.round(symmetrize(1 - embedded_sim), 12))

            metrics["dbcv"].append(clusterer.relative_validity_)
            metrics["trustworthiness"].append(
                trustworthiness(
                    dissim,
                    embedded_dissim,
                    n_neighbors=params["n_neighbors"],
                    metric="precomputed",
                )
            )
            metrics["density_preservation"].append(
                density_preservation_correlation(
                    dissim,
                    embedded_dissim,
                    n_neighbors=params["n_neighbors"],
                    metric="precomputed",
                )
            )

            module_labels.append(
                modules.module.map(lambda x: int(x.split("_")[1]) + 1).to_list()
            )

        consistency_scores = []
        for pair in combinations(module_labels, 2):
            consistency_scores.append(
                (
                    adjusted_mutual_info_score(pair[0], pair[1]),
                    adjusted_rand_score(pair[0], pair[1]),
                )
            )

        epsilon = 1e-12
        mean_ami = np.mean([score[0] for score in consistency_scores]) + epsilon
        mean_ari = np.mean([score[1] for score in consistency_scores]) + epsilon
        metrics["f1_consistency"].extend([mean_ami, mean_ari])

        trial.set_user_attr(
            "metrics",
            {key: float(np.array(value).mean()) for key, value in metrics.items()},
        )

        weights = {
            "dbcv": 0.05,
            "trustworthiness": 0.4,
            "density_preservation": 0.4,
            "f1_consistency": 0.15,
        }

        weighted_score = 0
        for metric, values in metrics.items():
            if values:
                if metric == "f1_consistency":
                    weighted_score += harmonic_mean(values) * weights[metric]
                else:
                    weighted_score += np.mean(values) * weights[metric]

        return weighted_score

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=n_startup_trials,
        seed=42,
    )

    pruner = optuna.pruners.HyperbandPruner(
        min_resource=n_warmup_steps,
        max_resource=n_trials,
        reduction_factor=reduction_factor,
    )

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    study.optimize(lambda trial: objective(trial, expr_df), n_trials=n_trials, n_jobs=1)

    best_params = study.best_params

    print(f"\nBest parameters found:")
    print(f"n_neighbors: {best_params['n_neighbors']}")
    print(f"n_components: {best_params['n_components']}")
    print(f"min_dist: {best_params['min_dist']}\n")
    print(f"Best value: {study.best_value}")
    print("Best trial metrics:\n", study.best_trial.user_attrs["metrics"])
    print()

    return best_params


def _sizes_from_fracs(fracs: List[float], n_genes: int, floor: int) -> List[int]:
    return sorted({max(floor, round(frac * n_genes)) for frac in fracs})


def _check_knowledge_args(gene_network: Optional[pd.DataFrame], gene_pathway_dict: Optional[Dict[str, Set[str]]]) -> None:
    if gene_network is None and gene_pathway_dict is None:
        raise ValueError(
            "knowledge_score optimization requires gene_network or gene_pathway_dict."
        )


def _print_grid_summary(title: str, all_results: List[Dict[str, Any]], best_params: Optional[Dict[str, Any]], best_score: float) -> None:
    print(f"\n----- {title} grid search (cheap re-condensing sweep, src) -----")
    print(
        pd.DataFrame(all_results)
        .sort_values("knowledge_score", ascending=False)
        .head(10)
        .to_string(index=False)
    )
    print(f"----- Best params: {best_params}")
    print(f"----- Best score: {best_score}\n")


def optimize_cgm_clustering_params(
    expr_df: pd.DataFrame,
    best_umap_params: Dict[str, int],
    min_clust_size_lst: Optional[List[int]] = None,
    min_samples_lst: List[int] = list(range(1, 6)),
    gene_network: Optional[pd.DataFrame] = None,
    gene_pathway_dict: Optional[Dict[str, Set[str]]] = None,
    hdbscan_n_jobs: int = -1,
) -> dict:
    """
    CGM parameter search: cheap re-condensing sweep.

    n_neighbors/n_components/min_dist come fixed from `best_umap_params`.
    For each candidate `min_samples`, `get_clust_hierarchy()` is called on
    ONE UMAP embedding (fit once for the whole sweep with these fixed
    parameters) and fits ONE HDBSCAN tree, then that tree is
    re-condensed at every candidate `min_cluster_size` via
    `recondense_labels()`, with no further MST refits.

    `min_clust_size_lst`, if not supplied, defaults to gene-count-relative
    fractions (3/5/7/10% of the genes, at least 5).

    Only `knowledge_score` is supported: DBCV (`relative_validity_`) is tied
    to a specific HDBSCAN `.fit()` call and is not recomputable for a
    re-condensed labeling without a fresh fit.
    """
    # TODO: this scores HDBSCAN-native labels (-1 ignored), while with
    # cgm_noise_filter == "none" the CGM output assigns every
    # gene by soft argmax, so the optimized and the output labelings differ.
    _check_knowledge_args(gene_network, gene_pathway_dict)

    if min_clust_size_lst is None:
        min_clust_size_lst = _sizes_from_fracs(
            [0.03, 0.05, 0.07, 0.10], expr_df.shape[1], 5
        )

    n_neighbors = best_umap_params["n_neighbors"]
    n_components = best_umap_params["n_components"]
    min_dist = best_umap_params["min_dist"]

    best_score = -np.inf
    best_params = None
    all_results = []
    knowledge_cache = (
        precompute_knowledge_edges(expr_df, gene_network)
        if gene_network is not None and gene_pathway_dict is None
        else None
    )

    # The embedding does not depend on min_samples: fit once for the sweep
    # (random_state 42 = get_clust_hierarchy's default).
    _, embedding = _embed_dissim(expr_df, n_neighbors, min_dist, n_components, 42)

    for min_samples in min_samples_lst:
        clusterer = get_clust_hierarchy(
            expr_df,
            min_samples=min_samples,
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            base_min_cluster_size=min(min_clust_size_lst),
            hdbscan_n_jobs=hdbscan_n_jobs,
            embedding=embedding,
        )
        for min_clust_size in min_clust_size_lst:
            labels = recondense_labels(clusterer, min_cluster_size=min_clust_size)

            knowledge = compute_knowledge_score(
                expr_df,
                labels,
                gene_network=gene_network,
                gene_pathway_dict=gene_pathway_dict,
                knowledge_cache=knowledge_cache,
            )

            all_results.append(
                {
                    "min_clust_size": min_clust_size,
                    "min_samples": min_samples,
                    "knowledge_score": knowledge,
                }
            )

            if knowledge > best_score:
                best_score = knowledge
                best_params = {
                    "min_clust_size": min_clust_size,
                    "min_samples": min_samples,
                    "n_neighbors": n_neighbors,
                    "n_components": n_components,
                    "min_dist": min_dist,
                }

    _print_grid_summary("CGM", all_results, best_params, best_score)

    return best_params


def optimize_fgm_clustering_params(
    expr_df: pd.DataFrame,
    best_umap_params: Dict[str, int],
    min_clust_size_lst: List[int],
    min_samples_lst: List[int] = list(range(1, 4)),
    merge_size_multiplier_lst: List[float] = [1.5, 2.0, 2.5, 3.0],
    gene_network: Optional[pd.DataFrame] = None,
    gene_pathway_dict: Optional[Dict[str, Set[str]]] = None,
    hdbscan_n_jobs: int = -1,
) -> Tuple[dict, float]:
    """
    FGM parameter search: cheap re-condensing sweep, jointly over
    (min_cluster_size, min_samples, merge_size_multiplier).

    Embedding params (n_neighbors/n_components/min_dist) come fixed from
    `best_umap_params`. The UMAP is fit once per call and
    `get_clust_hierarchy()` is called per `min_samples` value on that
    embedding (one HDBSCAN tree each); both `min_clust_size` (the fine cut) and
    `merge_size_multiplier` (via `min_clust_size * merge_size_multiplier`,
    the coarse "merge" cut) are swept via cheap `recondense_labels()` calls
    against that one tree.

    Scoring uses the *merged* labeling (the coarse cut mapped onto the fine
    clusters by `_union_of_fine`, as in `get_fgms_from_cgm`), not the fine one: the merged
    labeling is what `run_nested_wgcna.py`'s `extract_and_save_fgms()`
    actually writes to the output GMT files, so parameter selection optimizes
    what is produced.

    Returns `(best_params, best_score)` — the score is returned (unlike
    `optimize_cgm_clustering_params`) because `get_fgms_from_cgm()` calls
    this once per shortlisted `(corr_thr[, k])` candidate (see
    `select_confounder_params()`) and needs to compare scores *across*
    candidates, not just pick the best params within one.
    """
    # TODO: this scores the hard HDBSCAN labels merged by `_union_of_fine`, which
    # is exactly the output with noise_method="hdbscan"; with "kme" the output
    # labels are soft-assigned plus the kME noise step, so they differ.
    _check_knowledge_args(gene_network, gene_pathway_dict)

    n_neighbors = best_umap_params["n_neighbors"] // 2 + 1
    n_components = best_umap_params["n_components"]
    min_dist = best_umap_params["min_dist"]

    best_score = -np.inf
    best_params = None
    all_results = []
    knowledge_cache = (
        precompute_knowledge_edges(expr_df, gene_network)
        if gene_network is not None and gene_pathway_dict is None
        else None
    )

    # The embedding does not depend on min_samples: fit once for the sweep
    # (random_state 42 = get_clust_hierarchy's default).
    _, embedding = _embed_dissim(expr_df, n_neighbors, min_dist, n_components, 42)

    for min_samples in min_samples_lst:
        clusterer = get_clust_hierarchy(
            expr_df,
            min_samples=min_samples,
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            base_min_cluster_size=min(min_clust_size_lst),
            hdbscan_n_jobs=hdbscan_n_jobs,
            embedding=embedding,
        )
        for min_clust_size in min_clust_size_lst:
            fine_labels = recondense_labels(clusterer, min_cluster_size=min_clust_size)
            for multiplier in merge_size_multiplier_lst:
                coarse_size = max(
                    min_clust_size + 1, round(min_clust_size * multiplier)
                )
                # same merge as get_fgms_from_cgm: the coarse cut only votes
                merged_labels = _union_of_fine(
                    fine_labels,
                    recondense_labels(clusterer, min_cluster_size=coarse_size),
                )

                knowledge = compute_knowledge_score(
                    expr_df,
                    merged_labels,
                    gene_network=gene_network,
                    gene_pathway_dict=gene_pathway_dict,
                    knowledge_cache=knowledge_cache,
                )

                all_results.append(
                    {
                        "min_clust_size": min_clust_size,
                        "min_samples": min_samples,
                        "merge_size_multiplier": multiplier,
                        "coarse_size": coarse_size,
                        "knowledge_score": knowledge,
                    }
                )

                if knowledge > best_score:
                    best_score = knowledge
                    best_params = {
                        "min_clust_size": min_clust_size,
                        "min_samples": min_samples,
                        "merge_size_multiplier": multiplier,
                        "n_neighbors": n_neighbors,
                        "n_components": n_components,
                        "min_dist": min_dist,
                    }

    _print_grid_summary("FGM", all_results, best_params, best_score)

    return best_params, best_score


def select_confounder_params(
    module_expr_df: pd.DataFrame,
    core: Optional[List[str]] = None,
    corr_thr_lst: List[float] = [0.95, 0.9, 0.85, 0.8, 0.75],
    method: str = "spearman",
    min_ings_frac: float = 0.01,
    n_bootstrap: int = 3,
    final_top_k: int = 3,
    random_state: int = 42,
    input_is_log: bool = False,
) -> List[Dict]:
    """
    Tier 1 of the (corr_thr[, k]) selection for the confounder regression
    (`src/confounder_regression.py`): INGS thresholded per `corr_thr` from one
    eigengene correlation pass (see `select_ings()`), then bootstrap stability of that INGS set. The caller (`get_fgms_from_cgm()`)
    is responsible for Tier 2 (scoring the shortlist by `knowledge_score`).

    INGS sets are nested in `corr_thr`, so equal sets are kept once (the larger
    `corr_thr`). Candidates with < 2 INGS or an INGS fraction of the CGM's genes
    below `min_ings_frac` are dropped. Stability = mean Jaccard overlap between
    the full-data INGS and the INGS of `n_bootstrap` metacell resamples (an
    empty/failed resample counts as 0).

    Returns up to `final_top_k` dicts, best-first (stability descending, ties to
    the larger `corr_thr`), with keys `corr_thr`, `ings`, `n_ings`,
    `ings_fraction`, `stability`. Returns `[]` if no candidate is valid; callers
    fall back to the raw module expression.
    """
    n_genes = module_expr_df.shape[1]
    n_samples = module_expr_df.shape[0]
    candidates, seen = [], set()

    # The correlations do not depend on corr_thr: compute once, threshold per candidate.
    try:
        full_cor = _ings_correlations(module_expr_df, core, method, input_is_log)
        full_err = None
    except Exception as e:
        full_err = e

    for corr_thr in sorted(corr_thr_lst, reverse=True):
        if full_err is not None:
            print(f"Exception {full_err} occurred with corr_thr={corr_thr}")
            continue
        ings = _ings_from_correlations(full_cor, corr_thr)
        if len(ings) < 2 or len(ings) / n_genes < min_ings_frac:
            continue
        key = frozenset(ings)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "corr_thr": corr_thr,
                "ings": ings,
                "n_ings": len(ings),
                "ings_fraction": len(ings) / n_genes,
            }
        )

    if not candidates:
        print("Confounder Tier 1: no valid corr_thr candidates found.")
        return []

    # Every candidate sees the same resamples (same seed), so correlate each once.
    rng = np.random.default_rng(random_state)
    boot_cors = []
    for _ in range(n_bootstrap):
        resampled = module_expr_df.iloc[rng.choice(n_samples, size=n_samples)]
        try:
            boot_cors.append(_ings_correlations(resampled, core, method, input_is_log))
        except Exception:
            boot_cors.append(None)  # failed resample -> empty INGS

    for c in candidates:
        base = set(c["ings"])
        jaccards = []
        for cor in boot_cors:
            boot = set(_ings_from_correlations(cor, c["corr_thr"]))
            jaccards.append(len(base & boot) / len(base | boot))
        c["stability"] = float(np.mean(jaccards))

    candidates.sort(key=lambda c: (-c["stability"], -c["corr_thr"]))
    shortlist = candidates[:final_top_k]

    print("\n----- Confounder Tier 1 shortlist (bootstrap stability) -----")
    for c in shortlist:
        print(
            f"corr_thr={c['corr_thr']}, n_INGS={c['n_ings']}, "
            f"ings_fraction={c['ings_fraction']:.4f}, stability={c['stability']:.4f}"
        )
    print()

    return shortlist
