import pandas as pd
from pathlib import Path

def maturity_summary_report(
    df: pd.DataFrame,
    *,
    id_col: str = "LoanSequenceNumber",
    maturity_col: str = "MaturityDate",
    cutoff_year: int = 2025,
    output_dir: str = "Outputs/reports/data_analysis",
    filename: str = "maturity_summary_report.csv",
) -> pd.DataFrame:
   
    # Ensure columns exist
    if maturity_col not in df.columns or id_col not in df.columns:
        raise KeyError(f"'{maturity_col}' or '{id_col}' not found in DataFrame columns.")

    # Compute maturity year and metrics
    df = df.copy()
    df[maturity_col] = pd.to_datetime(df[maturity_col], errors="coerce")
    df["MaturityYear"] = df[maturity_col].dt.year

    most_common_year = df["MaturityYear"].mode()[0] if not df["MaturityYear"].empty else None
    avg_year = df["MaturityYear"].mean() if not df["MaturityYear"].empty else None

    mask = df[maturity_col] <= pd.Timestamp(f"{cutoff_year}-12-31")
    num_loans_cutoff = df.loc[mask, id_col].nunique()

    # Summary
    summary = pd.DataFrame({
        "Metric": [
            "Most common maturity year",
            "Average maturity year",
            f"Number of loans maturing by {cutoff_year}"
        ],
        "Value": [
            most_common_year,
            round(avg_year, 2) if avg_year else None,
            num_loans_cutoff
        ]
    })

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path / filename, index=False)


    return summary
