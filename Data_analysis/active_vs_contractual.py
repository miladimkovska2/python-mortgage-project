import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path  

def plot_active_vs_contractual_loans(merged: pd.DataFrame, start_year: int = 2010, end_year: int = 2025):

    
    # Aggregate to loan level
    loan_level = (
        merged.groupby("LoanSequenceNumber", as_index=False)
        .agg(
            MaturityDate=("MaturityDate", "first"),
            ZeroBalanceEffectiveDate=("ZeroBalanceEffectiveDate", "min")))

    # Extract years
    loan_level["MaturityYear"] = loan_level["MaturityDate"].dt.year
    loan_level["ZBYear"] = loan_level["ZeroBalanceEffectiveDate"].dt.year


    # Calculate yearly active and contractual loans
    rows = []
    for y in range(start_year, end_year + 1):
        active = loan_level[
            (loan_level["MaturityYear"] > y) &
            (loan_level["ZBYear"].isna() | (loan_level["ZBYear"] > y))
        ]["LoanSequenceNumber"].nunique()

        contractual = loan_level[
            loan_level["MaturityYear"] > y
        ]["LoanSequenceNumber"].nunique()

        rows.append({"Year": y, "ActiveLoans": active, "ContractualActive": contractual})

    combined = pd.DataFrame(rows)


    # Plot
    bar_width = 0.4
    x = np.arange(len(combined["Year"]))
    plt.figure(figsize=(12,6))
    plt.bar(x - bar_width/2, combined["ActiveLoans"], width=bar_width, color="#2f3b69", label="Observed active")
    plt.bar(x + bar_width/2, combined["ContractualActive"], width=bar_width, color="#c197d2", label="Contractual active")

    plt.xticks(x, combined["Year"], rotation=45)
    plt.xlabel("Year")
    plt.ylabel("Number of active loans")
    plt.title("Active Loans per Year: Observed vs Contractual (2010–2025)")
    plt.legend()
    plt.tight_layout()


    fig_dir = Path("Outputs/Figures/data_analysis")
    fig_dir.mkdir(parents=True, exist_ok=True)

    #Save plot
    save_path = fig_dir / "active_loans_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return combined
