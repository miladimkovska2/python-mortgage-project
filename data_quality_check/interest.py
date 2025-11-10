import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict


def analyze_rate_modification_consistency(
    df: pd.DataFrame,
    *,
    id_col: str = "LoanSequenceNumber",
    rate_col: str = "CurrentInterestRate",
    mod_col: str = "ModificationFlag",
    output_dir: str = "Outputs/reports",
    filename: str = "rate_modification_summary.png",
) -> Dict[str, int]:
   

    df = df.copy()

    # Identify interest rate changes 
    df["prev_rate"] = df.groupby(id_col)[rate_col].shift()
    df["changed_seq"] = df["prev_rate"].notna() & (df[rate_col] != df["prev_rate"])
    n_varying_rates = int(df.groupby(id_col)["changed_seq"].any().sum())

    # Identify modified loans
    mod_per_loan = df.groupby(id_col)[mod_col].apply(lambda x: (x == "Y").any())
    n_modified_loans = int(mod_per_loan.sum())

    #Loans with both modifications and varying rates
    loans_with_both = df.groupby(id_col)["changed_seq"].any()
    loans_with_both = loans_with_both[loans_with_both.index.isin(mod_per_loan[mod_per_loan].index)]
    n_both = int(loans_with_both.sum())



    # Prepare summary table
    results = {
        "Loans with Interest Rate Changes": n_varying_rates,
        "Loans with Modifications": n_modified_loans,
        "Loans with Both": n_both,
    }

    summary_df = pd.DataFrame(results.items(), columns=["Metric", "Count"])


    # Save table
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 1.8))
    ax.axis("off")
    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, loc="center")
    fig.savefig(output_path / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

    
    return results
