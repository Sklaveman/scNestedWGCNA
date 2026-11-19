import argparse

import pandas as pd

from src.normalization import (
    groups_from_metacell_names,
    normalize_metacell_counts,
    write_normalization_outputs,
)
from typing import List, Optional


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cell-type-aware poscounts normalization of summed metacell counts."
    )
    parser.add_argument(
        "--input", required=True, help="CSV of summed counts (metacells x genes)."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Normalized CSV path; Y_size_factors.csv and Y_group_scales.csv are written next to it.",
    )
    parser.add_argument(
        "--groups-file",
        default=None,
        help="CSV with columns metacell, group. Default: group = metacell name before the last '_'.",
    )
    parser.add_argument("--min-mean", type=float, default=1.0)
    parser.add_argument(
        "--exclude-regex",
        default=None,
        help="Case-insensitive regex of genes left out of the factor estimation "
        "(suggested: normalization.DOMINANT_GENES_REGEX).",
    )
    parser.add_argument("--target-total", type=float, default=1e4)
    parser.add_argument("--log1p", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_arguments(argv)
    counts = pd.read_csv(args.input, index_col=0)
    if args.groups_file:
        g = pd.read_csv(args.groups_file)
        groups = pd.Series(
            g["group"].astype(str).to_numpy(), index=g["metacell"].astype(str)
        )
    else:
        groups = groups_from_metacell_names(counts.index)
    norm, sf_df, scale = normalize_metacell_counts(
        counts, groups, args.min_mean, args.exclude_regex, args.target_total, args.log1p
    )
    write_normalization_outputs(norm, sf_df, scale, args.output)
    print(f"Wrote {args.output} ({norm.shape[0]} metacells x {norm.shape[1]} genes)")
    print("Group scales:\n" + scale.round(3).to_string())


if __name__ == "__main__":
    main()
