import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_estimated_ltv_trend(
    perf: pd.DataFrame,
    *,
    start_year: int = 2010,
    end_year: int = 2025,
    fig_dir: str = "Outputs/Figures/data_analysis",
    fig_filename: str = "estimated_ltv_trend.png",
) -> pd.DataFrame:
   
    df = perf.copy()
    df["Year"] = df["MonthlyReportingPeriod"].dt.year

    yearly_ltv = (df.groupby("Year", as_index=False)["EstimatedLTV"].mean().dropna())

    yearly_ltv = yearly_ltv[(yearly_ltv["Year"] >= start_year) & (yearly_ltv["Year"] <= end_year)]

    # Plot
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(
        yearly_ltv["Year"],
        yearly_ltv["EstimatedLTV"],
        color="#2f3b69",         
        marker="o",
        linewidth=2,
        label="Average Estimated LTV")

    plt.xlabel("Year")
    plt.ylabel("Estimated LTV")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    fig_path = Path(fig_dir) / fig_filename
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


    return yearly_ltv
