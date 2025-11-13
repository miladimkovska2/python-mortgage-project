from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path


def run_consistency_checks(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    id_col: Optional[str] = None,
    date_col: Optional[str] = None,
    cross_field_tuple: Optional[Tuple[str, str, str]] = None,
    rate_col: Optional[str] = None,
    mod_col: Optional[str] = None,
    output_dir: str = "Outputs/reports/Quality_Results",
    filename: str = "consistency_report.csv",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
   

    results: Dict[str, int | float] = {}


    # Temporal monotonicity check 
    if id_col and date_col and {id_col, date_col}.issubset(df2.columns):
        df_temp = df2[[id_col, date_col]].copy()
        df_temp["prev_date"] = df_temp.groupby(id_col)[date_col].shift(1)
        results["Temporal_Monotonicity_Violations"] = int((df_temp[date_col] < df_temp["prev_date"]).sum())


    # IDs map check 
    if id_col and id_col in df1.columns and id_col in df2.columns:
       ids_1 = set(df1[id_col].dropna())
       ids_2 = set(df2[id_col].dropna())
       results["ID_Difference_Count"] = len(ids_1.symmetric_difference(ids_2))



    # Cross-field rule check 
    if cross_field_tuple and all(f in df2.columns for f in cross_field_tuple):
        f1, f2, ref = cross_field_tuple
        tmp = df2[[f1, f2, ref]].copy()
        mask = (tmp[f1].notna()) & (tmp[f2] == "not_applicable") & (tmp[ref] != 0)
        results["Cross_Field_Violations"] = int(mask.sum())


    # Interest-rate modification consistency check 
    if id_col and rate_col and mod_col and {id_col, rate_col, mod_col}.issubset(df2.columns):
      df2 = df2.copy()
      df2["prev_rate"] = df2.groupby(id_col)[rate_col].shift()
      df2["rate_changed"] = df2["prev_rate"].notna() & (df2[rate_col] != df2["prev_rate"])


    # Identify per-loan flags
    rate_changed = df2.groupby(id_col)["rate_changed"].any()
    modified = df2.groupby(id_col)[mod_col].apply(lambda x: (x == "Y").any())

    n_varying_rates = int(rate_changed.sum())
    n_modified_loans = int(modified.sum())
    n_both = int((rate_changed & modified).sum())

    results["Loans_with_Rate_Changes"] = n_varying_rates
    results["Loans_with_Modifications"] = n_modified_loans
    results["Loans_with_Both"] = n_both


    # Drop loans with varying interest rate
    remove_ids = rate_changed.index[modified]
    df2 = df2[~df2[id_col].isin(remove_ids)].copy()
    if id_col in df1.columns:
        df1 = df1[~df1[id_col].isin(remove_ids)].copy()



    # Compute consistency score
    violation_keys = [
    "Temporal_Monotonicity_Violations",
    "ID_Difference_Count",
    "Cross_Field_Violations",
    "Loans_with_Rate_Changes"]

    total_violations = sum(results[k] for k in violation_keys if k in results)


    n_loans = df1[id_col].nunique()
    n_rows = len(df2)

    row_level_checks = ["Temporal_Monotonicity_Violations", "Cross_Field_Violations"]
    loan_level_checks = ["Loans_with_Rate_Changes"]
    dataset_level_checks = ["ID_Difference_Count"]

    denom = (
    len(row_level_checks) * n_rows
    + len(loan_level_checks) * n_loans
    + len(dataset_level_checks))


    results["Consistency_Score"] = (round(1 - total_violations / denom, 3)
    if denom and denom > 0 else np.nan)

# Save report
    output_path = Path.cwd() / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results.items(), columns=["Check", "Value"]).to_csv(
    output_path / filename, index=False)

    for col in ["prev_rate", "rate_changed"]:
        if col in df2.columns:
            df2 = df2.drop(columns=col)

    return df1, df2, results

