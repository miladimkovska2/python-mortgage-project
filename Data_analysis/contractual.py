import pandas as pd
from pathlib import Path

def loan_summary_report(
    df: pd.DataFrame,
    *,
    id_col: str = "LoanSequenceNumber",
    report_date_col: str = "MonthlyReportingPeriod",
    maturity_col: str = "MaturityDate",
    output_dir: str = "Outputs/reports/data_analysis",
    filename: str = "loan_summary_report.csv",
) -> pd.DataFrame:


    #  Compute summary metrics 
    num_loans = df[id_col].nunique()
    first_month = df[report_date_col].min()
    last_month = df[report_date_col].max()
    earliest_maturity = df[maturity_col].min()
    latest_maturity = df[maturity_col].max()


    # Build summary table 
    summary = pd.DataFrame({
        "Metric": [
            "Number of distinct Loan IDs",
            "First month (Reporting)",
            "Last month (Reporting)",
            "Earliest maturity date",
            "Latest maturity date"
        ],
        "Value": [
            num_loans,
            first_month,
            last_month,
            earliest_maturity,
            latest_maturity
        ]
    })

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path / filename, index=False)

    return summary
