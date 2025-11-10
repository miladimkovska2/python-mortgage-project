from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_categorical_dtype


def check_representativeness(
    df: pd.DataFrame,
    *,
    categorical_cols: Iterable[str] = (),
    binary_cols: Iterable[str] = (),
    numeric_cols: Iterable[str] = (),
    dom_threshold: float = 0.95,
    low_bin_threshold: float = 0.05,
    n_bins: int = 10,
    include_na_in_shares: bool = False,
    output_dir: str = "Outputs/reports/Quality_Results",
    image_name: str = "representativeness_report.png",
) -> dict[str, pd.DataFrame]:
   
    records: list[Dict[str, Any]] = []

    # CATEGORICAL
    for col in categorical_cols:
        if col not in df.columns:
            continue

        # Exclude placeholder values that are not real categories
        if is_categorical_dtype(df[col]):
          s = df[col].astype(str).replace(["not_applicable", "not_modified"], np.nan)
        else:
          s = df[col].replace(["not_applicable", "not_modified"], np.nan)

        try:
            vc = s.value_counts(normalize=True, dropna=not include_na_in_shares) * 100
            n_levels = vc.shape[0]
            top_cat = vc.index[0] if n_levels > 0 else np.nan
            top_share = float(vc.iloc[0]) if n_levels > 0 else np.nan
        except Exception:
            n_levels, top_cat, top_share = np.nan, np.nan, np.nan

        records.append({
            "column": col,
            "type": "categorical",
            "n_levels": int(n_levels) if pd.notna(n_levels) else np.nan,
            "top_category": top_cat,
            "top_share_%": round(top_share, 2) if pd.notna(top_share) else np.nan
        })
    
    # BINARY
    for col in binary_cols:
        if col not in df.columns:
            continue
        s = df[col].replace({"Y": 1, "N": 0, "y": 1, "n": 0}).copy()
        s = pd.to_numeric(s, errors="coerce")

        if s.notna().sum() == 0:
            records.append({
                "column": col,
                "type": "binary",
                "share_1_%": np.nan,
                "share_0_%": np.nan,
                "minority_share_%": np.nan
            })
            continue

        vc = s.value_counts(normalize=True, dropna=True)
        p1 = float(vc.get(1, 0.0))
        p0 = float(vc.get(0, 0.0))
        minority_share = min(p0, p1) * 100

        records.append({
            "column": col,
            "type": "binary",
            "share_1_%": round(p1 * 100, 2),
            "share_0_%": round(p0 * 100, 2),
            "minority_share_%": round(minority_share, 2)
        })

    # NUMERIC
    for col in numeric_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            records.append({
                "column": col,
                "type": "numeric",
                "bins": np.nan,
                "lowest_bin_%": np.nan
            })
            continue

        try:
            binned = pd.qcut(s, q=min(n_bins, max(2, s.nunique())), duplicates="drop")
            bin_counts = binned.value_counts(normalize=True, dropna=True).sort_index() * 100
            lowest_bin = float(bin_counts.min()) if not bin_counts.empty else np.nan
        except Exception:
            lowest_bin = np.nan

        records.append({
            "column": col,
            "type": "numeric",
            "bins": int(min(n_bins, s.nunique())),
            "lowest_bin_%": round(lowest_bin, 2) if pd.notna(lowest_bin) else np.nan
        })

    # Combine 
    summary = pd.DataFrame.from_records(records).sort_values(["type", "column"]).reset_index(drop=True)

    cat_tbl = summary.loc[summary["type"] == "categorical"].drop(columns="type", errors="ignore")
    bin_tbl = summary.loc[summary["type"] == "binary"].drop(columns="type", errors="ignore")
    num_tbl = summary.loc[summary["type"] == "numeric"].drop(columns="type", errors="ignore")

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, len(summary) * 0.5 + 1))
    ax.axis("off")
    ax.table(
        cellText=summary.drop(columns="type", errors="ignore").values,
        colLabels=summary.drop(columns="type", errors="ignore").columns,
        cellLoc="center",
        loc="center"
    )
    plt.tight_layout()
    plt.savefig(output_path / image_name, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "categorical": cat_tbl,
        "binary": bin_tbl,
        "numeric": num_tbl
    }


def compute_overall_representativeness_score(report_perf: dict, report_orig: dict) -> float:
    """
    Compute a single overall representativeness score across origination and performance datasets.
    Formula: 1 - (# poorly represented variables / total variables)
    """
    import numpy as np
    import pandas as pd

    combined_reports = [report_perf, report_orig]
    total_vars = 0
    well_represented = 0

    for report in combined_reports:
        for key, table in report.items():
            if not isinstance(table, pd.DataFrame) or table.empty:
                continue
            total_vars += len(table)

            if key == "categorical":
                well_represented += (table["top_share_%"] < 95).sum()
            elif key == "binary":
                well_represented += (table["minority_share_%"] >= 5).sum()
            elif key == "numeric":
                well_represented += (table["lowest_bin_%"] >= 5).sum()

    if total_vars == 0:
        return np.nan

    overall_score = round(1 - ((total_vars - well_represented) / total_vars), 3)
    return overall_score
