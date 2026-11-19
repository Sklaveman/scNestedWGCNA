# scNestedWGCNA Method Description

## 1. Project Summary

**scNestedWGCNA** is a Python pipeline for hierarchical, data-driven detection of gene co-expression modules ("gene programs") from single-cell RNA-seq data. It clusters genes (not cells) in two nested levels:

- **CGMs (Coarse-Grained Modules)** — broad, top-level gene clusters.
- **FGMs (Fine-Grained Modules)** — sub-modules discovered *within* each CGM after a within-module normalization step that removes the dominant shared signal, exposing finer co-expression structure that would otherwise be masked.

The repository contains two parallel implementations of this pipeline: the archived original (`data/old/src/`) and the current `src/` (**"Variant B"**, formerly `src/`), which is a targeted redesign of the same algorithm. This document describes **`src`** specifically.

### Why `src` exists

`src` was written in response to two concrete problems observed with the original pipeline:

1. **Benchmark quality gap.** On simulated benchmark datasets (1140, 3150, and low-depth "d1k" gene co-expression simulations with known ground-truth modules), the original pipeline only beat competing tools (hdWGCNA, GeneNMF) on the smallest (1140-gene) dataset, and underperformed on ARI/NMI/Fowlkes-Mallows for the larger/noisier ones.
2. **Speed and memory cost.** Parameter search in the original pipeline required a full UMAP + HDBSCAN refit for every point on a grid (min_cluster_size × min_samples), which became prohibitively slow (~45 minutes for a 3150-gene dataset) and memory-hungry (correlation-matrix spikes at 8 GB RAM / 8-core hardware).

`src` keeps the original's core mathematics (adjacency computation, DensMAP+HDBSCAN clustering, graph-degeneracy "core erosion") intact, replaces ImmFocus by a log-space per-gene regression normalization (Step 8), but restructures **how parameters are searched, how modules are merged, and how CGMs are processed**, and fixes several bugs that were silently disabling parts of the original pipeline (see §3).

### Relationship to the archived `data/old/src/`

| | `data/old/src` (original) | `src` (Variant B) |
|---|---|---|
| Core similarity/clustering math | DensMAP UMAP + HDBSCAN on sqrt(1−R²) dissimilarity | **Identical** — unchanged |
| FGM-stage normalization | ImmFocus: RobustScaler+MinMaxScaler → PCA eigengene → Spearman INGS → CVR filter → divide by INGS mean | **Log-space per-gene Regression Normalization** (`src/confounder_regression.py`): same eigengene/INGS selection, no CVR filter; each gene is regressed on the INGS mean (leave-one-out for INGS genes) in log space and the fitted effect, weighted by `|r|^k`, is subtracted (see Step 8). `src/GenFocus.py` is gone |
| Input normalization | `run_seacells.py` writes TP10K-normalized CSVs | `run_seacells.py` writes **raw summed counts** only; a separate, optional cell-type-aware poscounts step (`src/normalization.py`, Step 1b) normalizes them. No TP10K/TPM conversion exists in `src` |
| Parameter search strategy | Full refit per grid point (Optuna GridSampler) | **Fit once, re-condense many times** from a single HDBSCAN hierarchy |
| min_cluster_size grid | Fixed absolute counts (e.g. 60–120) | **Fractions of gene count** (scales across dataset sizes) |
| FGM merge step | Eigengene-correlation merge (`merge_fgms`), computed but **never saved to output** | Density-native re-condensing merge (`recondense_labels`), **actually written to output** |
| Optimization objective | `dbcv`, `pathway_enrichment` (broken — string mismatch), `silhouette` (unimplemented) | Only `knowledge_score` (pathway/GRN odds-ratio enrichment), fully wired end-to-end |
| Dominant-cluster / coverage penalties | Computed but never subtracted from the score (dead code) | **Removed** — the objective is `knowledge_score` alone |
| CGM-level parallelism (`--n-jobs`) | Declared but crashes (`TypeError`) — never implemented | Real `joblib` parallel dispatch, load-balanced, thread-count-aware |
| `(corr_thr, k)` selection | Heuristic: most restrictive `corr_thr` yielding ≥2 INGS (+ `CVR_thr`) | Two-tier, stability- and `knowledge_score`-driven search (see step 6) |

---

## 2. Major Computation Steps

The pipeline runs in two stages that use separate conda environments: **SEACells metacell aggregation**, then the **nested CGM/FGM extraction** (`run_nested_wgcna.py`), optionally followed by **validation** against ground truth or other tools.

### Step 1 — Metacell aggregation & data loading
Raw single-cell counts are aggregated into metacells via SEACells (`run_seacells.py`) and saved as RAW summed counts (metacells × genes) to the CSV path given by `--out-filename` (the metacell AnnData can also be saved with `--out-adataname`; the kernel space is set with `--build-kernel-on`, default `X_pca`). `run_seacells.py` applies no normalization of any kind (no TP10K, no gene-length correction, no per-metacell-size scaling) and no small-metacell filter, so the matrix must be normalized separately before Step 2 (Step 1b). The downstream steps assume normalized, linear-scale input: the FGM-stage normalization (Step 8) applies `log1p` itself (unless `--input-is-log` says the matrix is already log-transformed) and does not convert counts. On load, genes with zero expression across all metacells are dropped (`load_and_filter_data`).

### Step 1b — Cell-type-aware poscounts normalization (optional, separate step)
`src/normalization.py` (numpy/pandas only, so it also runs in `seacells_env`) normalizes the summed counts. It is available as `--out-norm-filename` in `run_seacells.py` and as a standalone CLI (`python run_metacell_normalization.py --input X.csv --output Y.csv`, run from the project root). Groups are the cell types (or cell type × batch) SEACells built the metacells in; for the standalone CLI they default to the metacell name before the last `_` (or come from `--groups-file`).
1. **Poscounts size factors** (DESeq2 `estimateSizeFactors(type="poscounts")`): per gene, the reference is the mean log count over all samples (zeros add 0 to the sum but count in the denominator); per sample, the log ratio to the reference is computed on genes with count > 0 only; the factor is exp(median of those ratios); factors are rescaled to geometric mean 1.
2. **Within each group**, poscounts factors are computed on that group's metacells alone (a one-metacell group gets factor 1), using only genes with mean count ≥ `--min-mean` (default 1.0) and, optionally, not matching `--exclude-regex` (case-insensitive, so human and mouse symbols are covered; `normalization.DOMINANT_GENES_REGEX` is a suggested pattern for mitochondrial, ribosomal, immunoglobulin and hemoglobin genes). The factors are estimated on those genes but applied to all genes.
3. **Between groups:** each group's counts are divided by its within-group factors and averaged into a profile; poscounts on the profiles (genes positive in every profile) gives one scale per group.
4. **Final factor** = within-group factor × group scale, rescaled to geometric mean 1. Output = counts / factor, multiplied by one constant so the median metacell total is `--target-total` (default 1e4, which keeps the numeric scale the pipeline expects); `--log1p` optionally log-transforms (off by default, since it changes ImmFocus behaviour).
Outputs: the normalized CSV, `<stem>_size_factors.csv` (metacell, group, size_factor) and `<stem>_group_scales.csv`. Validation: identical metacells in counts and groups, no NaN/negatives, no zero-total metacell, ≥ 50 genes for estimation (error) and ≥ 50 genes positive in every profile (warning), all factors finite and > 0.
Why: total-count factors are pulled by a few dominant genes whose share differs between cell types, which can create spurious gene–gene correlations; a median-based factor resists this. Measured so far: on the three simulated metacell sets the log factors correlate 0.998–0.9998 with plain poscounts and 0.985–0.997 with total counts, so differences to TP10K are small there; on a synthetic dataset with a composition-skewed group the method recovered the true depth with a group-level shift of ≤ 0.005 (log scale) versus 0.05–0.35 for total counts. Not yet tested: the effect on module recovery by the pipeline; the DESeq2 equivalence (DESeq2 is not installed here).

### Step 2 — Gene–gene dissimilarity computation
For any gene set being clustered, a soft-thresholded correlation adjacency is computed (`get_adjacency`): Spearman correlation → signed rescaling to [0,1] → raised to power α=2 (WGCNA-style soft thresholding). This is converted to a dissimilarity matrix `sqrt(1 − R²)`, symmetrized and cast to `float32`.

### Step 3 — UMAP embedding parameter optimization
An Optuna TPE search (`optimize_umap_params`, unchanged from the original) tunes DensMAP UMAP's `n_neighbors`, `n_components`, and `min_dist`. Each trial is scored across 5 random seeds by a composite objective: 40% trustworthiness, 40% density preservation, 15% cluster-label consistency (harmonic mean of AMI/ARI across seeds), 5% DBCV.

### Step 4 — CGM clustering-parameter optimization (cheap re-condensing sweep)
This is `src`'s central architectural change. Rather than refitting UMAP+HDBSCAN at every (min_cluster_size, min_samples) grid point, `get_clust_hierarchy()` fits **one** UMAP embedding and, for each candidate `min_samples`, **one** HDBSCAN mutual-reachability tree. In the parameter sweeps the dissimilarity and UMAP embedding are computed once per sweep call (per normalization candidate in the FGM sweep) and passed to `get_clust_hierarchy(..., embedding=...)`, so the UMAP is not refit per `min_samples`; labels are identical to refitting. `recondense_labels()` then re-extracts a flat clustering at every candidate `min_cluster_size` directly from that tree's `single_linkage_tree_`, at near-zero marginal cost. Each candidate labeling is scored by `knowledge_score` — an odds-ratio enrichment of co-expressed gene pairs against a supplied gene-regulatory-network/pathway graph. This score is the whole objective. `min_cluster_size` candidates are expressed as fractions of total gene count so the search scales across datasets of different sizes.

### Step 5 — CGM extraction
A final, proper UMAP+HDBSCAN fit is run at the chosen (or user-supplied) parameters (`get_cgms`). Soft cluster membership vectors (`hdbscan.all_points_membership_vectors`) assign each gene to its most likely module; there is no outlier-score cut. Noise handling is configurable (`--cgm-noise-filter`): `none` (default) puts every gene in its soft-argmax CGM; `hdbscan` marks HDBSCAN-native noise (`clusterer.labels_ == -1`) as `module_-1`. A graph-degeneracy "core erosion" pass (`errode_clusters` / `core_decomposition`) then strips each cluster down to its densely-connected core, producing a second, stricter `core_module` label alongside the soft `module` label.

**Gene pre-filter (`--prefilter-genes`, `topk_coexpression_filter`).** For each gene, the statistic is the mean of its `k` (default 5) largest Spearman correlations with the other genes. The null permutes each gene's metacell values independently (one full permuted matrix per permutation, so results do not depend on chunking) and correlates it with the unpermuted others; null statistics are pooled within quantile bins of the gene's zero fraction, upper-tail empirical p-values, Benjamini-Hochberg at `--prefilter-fdr` (0.1). Zero-variance genes are always noise. Caveats: pooling assumes similar-sparsity genes share a null; metacells are not independent (p-values slightly anticonservative); `k` must stay below the smallest module size to keep; with fewer than ~300 genes BH can reject only if several genes are strongly co-expressed. The removal is irreversible, and the `--cgm-min-clust-frac-*` fractions become relative to the filtered gene count. On the 1140 benchmark with shuffled-copy decoys it removed ~90% of pure-noise decoys at fdr 0.1 and 0% of real genes.

### Step 6 — Per-CGM normalization parameter selection (two-tier search)
For each CGM, before sub-clustering, the normalization hyperparameters `corr_thr` (and optionally `k`) are chosen instead of the original's fixed heuristic:
- **Tier 1 (bootstrap, `select_confounder_params()`):** one INGS selection per `corr_thr` in `--norm-corr-thr-grid` (default 0.95 … 0.5; equal INGS sets are kept once, with the larger `corr_thr`), then bootstrap-resampling of metacells and mean Jaccard overlap of the resampled INGS with the full-data INGS. Candidates are ranked by this stability (ties: larger `corr_thr`); the top 3 go on. There is no CVR filter and no cheap score any more.
- **Tier 2 (downstream quality):** each shortlisted candidate is passed through the full FGM clustering-parameter search (Step 7) and scored by `knowledge_score`; the overall best is kept jointly with its clustering parameters. Default regime: `k = --norm-k` (2). With `--tune-norm-k` the candidates are the shortlist × `--norm-k-grid` (default 0 1 2 3 4); `norm_k` is evaluated first and ties (strict `>`) fall back to it, so equal scores (incl. all zero) keep the default `k` and the more stable `corr_thr`. This is up to ~5× more clustering searches per CGM. `--tune-norm-k` needs `--optimize-fgm`. Without Tier 2 the best Tier 1 candidate with `k = --norm-k` is used.
- The chosen `(corr_thr, k, n_ings, knowledge_score)` per CGM is kept in `submodules_df.attrs["normalization"]` and written to `{prefix}_fgm_normalization.csv`.

### Step 7 — FGM clustering-parameter optimization
The same cheap re-condensing strategy from Step 4 (`optimize_fgm_clustering_params`) is applied within each CGM, jointly sweeping `min_cluster_size` (as a fraction of the CGM's own gene count), `min_samples`, and a `merge_size_multiplier`. Critically, scoring is done on the **merged** labeling (Step 9), not the fine one — because the merged labeling is what actually reaches the saved output files.

### Step 8 — FGM extraction
Using the winning normalization parameters, gene expression within the CGM is normalized (`regress_out_confounder()`, see below) and re-clustered with a final UMAP+HDBSCAN fit (`get_fgms_from_cgm`), producing fine-grained submodules with soft `module` and core-eroded `module_core` labels.

**Log-space per-gene Regression Normalization** (`src/confounder_regression.py`; positional numpy, so duplicate metacell labels from bootstrap resamples are harmless; no division anywhere). `X` = normalized linear-scale module expression (metacells × genes), `Y = log1p(X)` (skipped with `--input-is-log`):
0. **Log scale.** `Y = log1p(X)`.
1. **INGS selection (`select_ings`).** Eigengene via RobustScaler → MinMaxScaler → 1-component PCA on `Y` over the module's core genes (all genes if no core), sign oriented towards the mean of the scaled genes. `INGS` = genes with Spearman correlation to the eigengene `> corr_thr` (a subset of the core when given); fewer than 2 means the candidate is invalid.
2. **Confounder signal.** `f` = mean of `Y` over INGS (geometric mean on the linear scale), `p = len(INGS)`. Non-INGS genes use `f`; INGS genes use the leave-one-out `f_i = (p·f − Y_i)/(p − 1)`.
3. **Per-gene fit.** Centred `Fc_i`, `Yc_i`; slope `b_i = Σ Fc·Yc / Σ Fc²`, correlation `r_i`. A gene or confounder without variance is left unchanged.
4. **Correction.** `w_i = |r_i|^k`, `corrected_i = Y_i − w_i · b_i · Fc_i`. Gene means are kept; only the centred confounder part is removed. `k = 0` is full regression removal (every non-INGS gene ends with correlation 0 to `f`); the fraction of a gene's variance remaining is `1 − r²(2w − w²)`; larger `k` removes less. Downstream clustering is Spearman-based, so negative values are fine.

### Step 9 — Density-native FGM merging
`src` replaces the original's eigengene-correlation merge (`merge_fgms`, whose result was computed but silently discarded by the original's output-writing code) with `recondense_labels()`. This re-condenses the **same** single-linkage tree used for the fine FGM fit, but at a coarser `min_cluster_size = fine_min_cluster_size × merge_size_multiplier`. The coarse cut is used only as a vote: `_union_of_fine()` sends each fine FGM as a whole to the coarse cluster most of its genes fall into, so every merged FGM is an exact union of fine FGMs and there are never more merged than fine FGMs (the raw coarse cut can have more clusters, because HDBSCAN's excess-of-mass selection is not monotonic in `min_cluster_size`). `--fgm-noise-method hdbscan` uses the hard labels of both cuts (native `-1` = `module_-1`, no kME); `kme` (default) uses soft assignments for both (`recondense_soft_labels()`: argmax membership vector on a shallow copy of the clusterer with the coarse condensed tree).

### Step 9b — FGM-level noise calls on the merged FGMs (kME)
With `--fgm-noise-method kme` (default), `kme_noise()` tests each gene of a CGM against the eigengene (PC1 of rank-z-scored expression, leave-one-out) of its own **fine** FGM, computed on the CGM's **raw** expression — not the normalized one. The decoy numbers (median kME 0.52 vs 0.72 normalized, 0.07 vs 0.87 raw) were measured on ImmFocus-normalized data and were NOT re-measured for the regression normalization; keeping kME on raw expression is a design choice made on that evidence. Fine rather than merged eigengenes so that a gene of a small constituent is not judged against a merged PC1 dominated by a larger one. The p-value uses the exact null (a standardized permuted gene vs. a fixed standardized eigengene has correlation variance 1/(n-1)), BH runs per CGM over genes that are not already HDBSCAN-native noise in the merged labels, and a gene is noise if `qval > --kme-fdr` (0.05) or `kme < --kme-min` (0.3). Noise calls change the **merged** labels only (noise genes are labelled `module_-1`, one noise module shared by all CGMs, like the CGM-level noise, the genes removed by the pre-filter and the non-core genes of the `*_core` columns); the fine `module`/`module_core` columns and fine GMTs are never modified. Modules with fewer than 4 genes cannot be tested (labels kept). The test asks whether a gene is related to its module at all, not for FDR-controlled membership; `0.3` was chosen on one benchmark. `merged_module_core` is eroded on integer labels and then prefixed. Per-gene diagnostics (`kme`, `kme_qval`) are saved to `{prefix}_fgm_kme.csv`.

### Step 10 — Parallel per-CGM orchestration
`extract_all_fgms()` dispatches Steps 6–9 independently per CGM via a picklable worker (`_process_cgm`), either sequentially or in parallel via `joblib.Parallel(backend="loky")`. CGMs are processed largest-first for better load balancing, and each CGM's internal HDBSCAN thread count (`hdbscan_n_jobs`) is throttled against the number of parallel workers to avoid CPU oversubscription. A failure on any single CGM falls back to keeping that CGM's genes as one unrefined block (no FGM split, no kME step, the whole CGM as its `module_core`) rather than aborting the whole run; the exception type and message are printed and, if any CGM failed, written to `{prefix}_raw_failed_cgms.csv` (CGM, n_genes, error). If no valid INGS candidate exists for a CGM, the raw module expression is clustered and FGMs are still produced (this is not the failure path). The CLI also skips the extra UMAP fit on unnormalized expression that only serves the before/after plot (`get_fgms_from_cgm(..., return_raw_embedding=False)`).

### Step 11 — GMT export
Four GMT gene-set files are written per run: fine raw / fine filtered (dense-core-only) and merged raw / merged filtered — exposing both the pre- and post-merge module assignments, unlike the original pipeline which only ever wrote the fine assignments.

### Step 12 — Validation (optional, separate script)
`benchmarks/run_validation.py` (dev-only, not part of the installable package; structurally unchanged from the original, minus plotting utilities; GMT files are looked up in `results/modules/`, then `<data-dir>/gene_modules/`) benchmarks the resulting FGMs against a ground-truth GMT and/or other tools' modules (hdWGCNA, GeneNMF):
- **Intrinsic metrics** (`benchmarks/metrics.py`): average intra-module Pearson correlation, silhouette score, DBCV, Calinski-Harabasz index (computed on 50-PC gene vectors).
- **Supervised metrics**: Adjusted Rand Index, Normalized Mutual Information, Fowlkes-Mallows Index against ground-truth modules. `make_gene_vs_module_df` assigns each gene to the first module containing it (with a warning for overlapping sets such as GeneNMF molecular programs) and labels unassigned genes `noise_module`. `run_validation.py` loads a set named `module_-1` (the pipeline's noise module) as `noise_module`, so it is treated as noise in the intrinsic metrics and matches the ground truth's noise class in the supervised ones (other sets are loaded with `exclude_noise=False`).

---

## 3. Bugs Fixed Relative to the Original (`data/old/src`) Pipeline

- **`--optimize-*` flags could not be switched off** (`store_true` with default `True`): `src` makes them opt-in `store_true` flags that are off by default, with fixed UMAP/CGM defaults used instead (the gene network is then only needed for `--optimize-cgm` / `--optimize-fgm`).
- **`run_seacells.py` option types**: `--min-cells`/`--min-celltype-size` are ints and `--use-gpu` goes through a bool parser (a string `"False"` used to be truthy).
- **ImmFocus defects** (history, superseded by the regression normalization; duplicate-index crash in the bootstrap stability score, division by zero with fewer than 2 INGS, an extra entry in the factor vector, order-dependent non-leave-one-out normalization) were fixed by the earlier `immfocus` rewrite, which has since been replaced altogether.
- **`errode_clusters`** no longer crashes on singleton clusters or gaps in the integer labels; **`make_gene_vs_module_df`** with an empty module dict now labels every gene `noise_module`.
- **`n_jobs` crash**: the original declared a `--n-jobs` CLI flag and passed it into `extract_all_fgms()`, which didn't accept the parameter — any use of `--n-jobs` raised `TypeError`. `src` implements real parallelism (Step 10).
- **Discarded merge step**: the original computed an eigengene-correlation-based FGM merge but never used its output when writing GMT files. `src`'s re-condensing merge is written to dedicated `_merged` output files.
- **`optimization-score` string mismatch**: the original's CLI accepted `pathway_enrichment` but the optimizer internally checked for `knowledge_score`, so the flag silently fell through. `src` removes the `optimization_score` parameter entirely (`knowledge_score` is the only objective); the gene network (`--gene-network-file`) is needed only with `--optimize-cgm` / `--optimize-fgm`.
- **`run_seacells.py` missing CLI arg**: the original's `main()` referenced `args.out_filename`, which was never defined as an argparse argument. `src` adds `--out-filename`.

## 4. Known Scope Limitations of `src`

- **Star-shaped GRNs.** The simulated GRNs have few regulators (1140: 6 regulators for 1432 edges; 2150: 10 for 5397) and no regulator is also a target, so every known pair contains a regulator and `knowledge_score` mostly measures whether regulators land in the same module as their targets. The effect on the chosen parameters was not tested. (The 2150 GRN file originally used bare integer ids that matched no expression gene; it now uses `geneN` like the others, and the original is kept in `data/old/`.)
- **Confounder vs. biology.** The INGS signal is purely data-driven: it cannot tell a technical axis from a real biological program (e.g. an activation gradient), and both are subtracted.
- **`k` tuning.** It uses the same `knowledge_score`, which on star-shaped GRNs mostly measures regulator–target co-clustering, and tuning an extra parameter per CGM can overfit.
- **Heuristic weight, no benchmark.** `w = |r|^k` is a heuristic (`k = 0` is plain full regression), and no benchmark of the regression normalization against ImmFocus (1140 / 2150 / d1k) has been run yet.
- **Induced INGS correlation.** The factor is estimated from the same genes it is removed from, which can induce small negative residual correlations among INGS genes (leave-one-out reduces but does not remove this). The model is linear in log space per gene.
- **Parameter search ignores the noise steps.** `optimize_cgm_clustering_params` / `optimize_fgm_clustering_params` score HDBSCAN-native labels without the gene pre-filter/kME noise handling (documented TODO).
- **Reproducibility.** The FGM stage can differ between processes and between `--n-jobs 1` and `--n-jobs 2` despite fixed seeds (cause not investigated), and `get_fgms_from_cgm` still hard-codes `random_state=42`; there is no `--seed` option.
- **Only-noise CGMs.** `extract_all_fgms` raises a `ValueError` (`pd.concat([])`) if every CGM is noise.
- Only the `knowledge_score` optimization objective is supported for CGM/FGM parameter search — DBCV was dropped because it isn't cheaply recomputable from a re-condensed (rather than freshly-fit) HDBSCAN labeling, and `silhouette` was never implemented in either variant. This means `src` **requires** a gene regulatory network or pathway-membership file as ground-truth knowledge to score against; it cannot run in a fully unsupervised parameter-search mode.
- UMAP embedding-parameter search (`n_neighbors`/`n_components`/`min_dist`) is *not* covered by the cheap re-condensing trick — a genuinely different embedding is required per combination, so Step 3 remains as costly as in the original pipeline.
