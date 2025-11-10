import pandas as pd
from pathlib import Path
from typing import Optional

def completeness_score(
    df1: pd.DataFrame,
    df2: Optional[pd.DataFrame] = None,
    id_col: Optional[str] = None,
    date_col: Optional[str] = None,
    output_dir: str = "Outputs/reports/Quality_Results",
    filename: str = "completeness_report.csv",
    exclude_cols: Optional[list[str]] = None  
) -> float:
    

    exclude_cols = exclude_cols or []
    frames = [df1] if df2 is None else [df1, df2]

    # Type 1: Missing Values 
    missing = 0
    total_cells = 0
    for df in frames:
        cols_to_check = [c for c in df.columns if c not in exclude_cols]
        missing += df[cols_to_check].isna().sum().sum()
        total_cells += df[cols_to_check].shape[0] * df[cols_to_check].shape[1]


    # Type 2: Temporal Gaps 
    total_gaps = 0
    loans_with_gaps = []

    if df2 is not None and id_col and date_col in df2.columns:
        perf = df2[[id_col, date_col]].copy()
        perf[date_col] = pd.to_datetime(perf[date_col], errors="coerce")

        perf["period_int"] = perf[date_col].dt.year * 12 + perf[date_col].dt.month
        perf = perf.sort_values([id_col, "period_int"])
        perf["prev_period"] = perf.groupby(id_col)["period_int"].shift(1)
        perf["gap"] = perf["period_int"] - perf["prev_period"]

        total_gaps = int(perf.loc[perf["gap"] > 1, "gap"].sub(1).sum())
        loans_with_gaps = perf.loc[perf["gap"] > 1, id_col].unique().tolist()



    # Combine both completeness types 
    gap_cells = total_gaps * (df2.shape[1] - len(exclude_cols) if df2 is not None else 0)
    expected_cells = total_cells + gap_cells
    total_missing = missing + gap_cells
    total_missing_fraction = total_missing / expected_cells if expected_cells > 0 else 0
    completeness = 1 - total_missing_fraction



    # Save report 
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {
        "Total Cells": expected_cells,
        "Type 1 missing values": missing,
        "Type 2 gap loans": total_gaps,
        "Type 2 gap cells": gap_cells,
        "Completeness_Score": round(completeness, 6),
        "Loans_with_Gaps": len(loans_with_gaps)
    }

    pd.DataFrame(results.items(), columns=["Metric", "Value"]).to_csv(
        Path(output_dir) / filename, index=False
    )


    # Remove loans with temporal gaps
    if loans_with_gaps:
        df2 = df2[~df2[id_col].isin(loans_with_gaps)].copy()
        if id_col in df1.columns:
            df1 = df1[~df1[id_col].isin(loans_with_gaps)].copy()

    return round(completeness, 6), df1, df2
