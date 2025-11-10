import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict


def uniqueness_score(
    df1: pd.DataFrame,
    df2: Optional[pd.DataFrame] = None,
    id_cols_df1: Optional[List[str]] = None,
    id_cols_df2: Optional[List[str]] = None,
    output_dir: str = "Outputs/reports/Quality_Results",
    filename: str = "uniqueness_report.csv",
) -> float:
    
    results: Dict[str, int] = {}


    #Duplicates in df1
    if id_cols_df1:
        d1_dupes = df1.duplicated(subset=id_cols_df1).sum()
    else:
        d1_dupes = df1.duplicated().sum()
    results["duplicates_df1"] = int(d1_dupes)


    # Duplicates in df2 (if provided)
    if df2 is not None:
        if id_cols_df2:
            d2_dupes = df2.duplicated(subset=id_cols_df2).sum()
        else:
            d2_dupes = df2.duplicated().sum()
        results["duplicates_df2"] = int(d2_dupes)
    else:
        results["duplicates_df2"] = 0


    # Total duplicates across both datasets
    total_duplicates = (
        results["duplicates_df1"]
        + results["duplicates_df2"])


    # Total evaluated records
    total_records = len(df1) + (len(df2) if df2 is not None else 0)
    if total_records == 0:
        return 0.0


    #Compute final score
    score = 1 - (total_duplicates / total_records)
    
    
    # Save report 
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results.items(), columns=["Metric", "Value"]).to_csv(
        Path(output_dir) / filename, index=False
    )


    return round(score, 6)

