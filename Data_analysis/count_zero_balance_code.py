import pandas as pd
from pathlib import Path

def loan_termination_report(
    df: pd.DataFrame,
    *,
    id_col: str = "LoanSequenceNumber",
    date_col: str = "MonthlyReportingPeriod",
    termination_col: str = "ZeroBalanceCode",
    output_dir: str = "Outputs/reports/data_analysis",
    filename: str = "loan_termination_report.csv",
) -> pd.DataFrame:

    # Compute last record per loan 
    last_record = (
        df.sort_values(date_col)
          .groupby(id_col)
          .tail(1)
    )
    filtered = last_record[last_record[termination_col] != "not_applicable"]

    # Count terminated loans per ZeroBalanceCode
    summary = (
        filtered.groupby(termination_col)[id_col]
        .nunique()
        .reset_index()
        .rename(columns={id_col: "LoanCount"})
        .sort_values("LoanCount", ascending=False)
        .reset_index(drop=True)
    )

    # Save report 
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path / filename, index=False)

    return summary
