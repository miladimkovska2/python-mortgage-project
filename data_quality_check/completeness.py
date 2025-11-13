import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

def completeness_score(
    df1: pd.DataFrame,
    df2: Optional[pd.DataFrame] = None,
    id_col: Optional[str] = None,
    date_col: Optional[str] = None,
    output_dir: str = "Outputs/reports/Quality_Results",
    filename: str = "completeness_report.csv",
    first_period_after_year: Optional[int] = 2011,
    exclude_cols: Optional[list[str]] = None
) -> Tuple[float, pd.DataFrame, Optional[pd.DataFrame]]:
   
    exclude_cols = exclude_cols or []
    frames = [df1] if df2 is None else [df1, df2]

    # Type 1: Missing Values 
    missing = 0
    total_cells = 0
    for _df in frames:
        cols_to_check = [c for c in _df.columns if c not in exclude_cols]
        if cols_to_check:
            missing += _df[cols_to_check].isna().sum().sum()
            total_cells += _df.shape[0] * len(cols_to_check)


    # Type 2: Temporal Gaps 
    total_gaps = 0
    loans_with_gaps: list[str] = []
    loans_first_after_cutoff: list[str] = []

    if (
        df2 is not None
        and id_col is not None and date_col is not None
        and id_col in df2.columns and date_col in df2.columns
    ):
        perf = df2[[id_col, date_col]].copy()
        perf[date_col] = pd.to_datetime(perf[date_col], errors="coerce")

        # Build integer month 
        perf["period_int"] = perf[date_col].dt.year * 12 + perf[date_col].dt.month
        perf = perf.sort_values([id_col, "period_int"])
        perf["prev_period"] = perf.groupby(id_col)["period_int"].shift(1)
        perf["gap"] = perf["period_int"] - perf["prev_period"]

        # Count months missing 
        total_gaps = int(perf.loc[perf["gap"] > 1, "gap"].sub(1).sum())

        # Loans with any internal gap
        loans_with_gaps = (
            perf.loc[perf["gap"] > 1, id_col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
        )

        # Loans whose first reporting month is after the cutoff year
        if first_period_after_year is not None:
            first_dates = perf.groupby(id_col, dropna=True)[date_col].min().dropna()
            first_years = first_dates.dt.year
            loans_first_after_cutoff = (
                first_years[first_years > int(first_period_after_year)]
                .index.astype(str)
                .tolist()
            )

    # Combining missing values + implied gap "cells" 
    gap_cells = 0
    if df2 is not None:
        
        checked_cols_df2 = max(df2.shape[1] - len(exclude_cols), 0)
        gap_cells = total_gaps * checked_cols_df2

    expected_cells = total_cells + gap_cells
    total_missing = missing + gap_cells
    total_missing_fraction = (total_missing / expected_cells) if expected_cells > 0 else 0.0
    completeness = 1.0 - total_missing_fraction

    # Save report 
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {
        "Total Cells": int(expected_cells),
        "Type 1 missing values": int(missing),
        "Type 2 gap months": int(total_gaps),
        "Type 2 gap cells": int(gap_cells),
        "Loans_with_Gaps": int(len(loans_with_gaps)),
        "Cutoff_FirstPeriod_After_Year": (
            int(first_period_after_year) if first_period_after_year is not None else None
        ),
        "Loans_FirstPeriod_After_Cutoff": int(len(loans_first_after_cutoff)),
        "Completeness_Score": round(float(completeness), 6),
    }
    pd.DataFrame(results.items(), columns=["Metric", "Value"]).to_csv(
        Path(output_dir) / filename, index=False
    )

    # Drop BOTH sets: gaps + after-cutoff 
    to_drop = set(loans_with_gaps) | set(loans_first_after_cutoff)
    if to_drop:
        if df2 is not None and id_col in df2.columns:
            df2 = df2[~df2[id_col].astype(str).isin(to_drop)].copy()
        if id_col in df1.columns:
            df1 = df1[~df1[id_col].astype(str).isin(to_drop)].copy()

    return round(float(completeness), 6), df1, df2
