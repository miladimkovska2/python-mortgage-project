import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path


def plot_correlation_matrix(
    perf: pd.DataFrame,
    orig: pd.DataFrame,
    *,
    fig_dir: str = "Outputs/Figures/data_analysis",
    fig_filename: str = "correlation_matrix.png",
) -> pd.DataFrame:
    
    BLUE, GREY, PURPLE = "#2f3b69", "#9f9f9f", "#c197d2"
    cmap = LinearSegmentedColormap.from_list("blue_grey_purple", [BLUE, GREY, PURPLE])

    first_report = (
        perf.groupby("LoanSequenceNumber", as_index=False)
        .agg(FirstReportingPeriod=("MonthlyReportingPeriod", "min"))
    )

    merged = (
        perf.merge(orig[["LoanSequenceNumber", "MaturityDate"]],
                   on="LoanSequenceNumber", how="left")
        .merge(first_report, on="LoanSequenceNumber", how="left"))

    # Compute LoanAge (months since first reporting)
    merged["LoanAge"] = ((merged["MonthlyReportingPeriod"] - merged["FirstReportingPeriod"]).dt.days / 30)

   
    # Select and correlate key variables
    corr_vars = merged[["CurrentActualUPB", "EstimatedLTV", "LoanAge"]].copy()
    corr = corr_vars.corr(numeric_only=True)

   
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        square=True
    )

    plt.title("Correlation Matrix of Loan Characteristics", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fig_path = Path(fig_dir) / fig_filename
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


    return corr
