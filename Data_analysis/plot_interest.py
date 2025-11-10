import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_interest_rate_trend(
    perf: pd.DataFrame,
    *,
    start_year: int = 2010,
    end_year: int = 2025,
    fig_dir: str = "Outputs/Figures/data_analysis",
    fig_filename: str = "interest_rate_trend.png",
) -> pd.DataFrame:
   
    df = perf.copy()
    df["Year"] = df["MonthlyReportingPeriod"].dt.year

    yearly_rate = (
        df.groupby("Year", as_index=False)["CurrentInterestRate"]
        .mean()
        .dropna())

    yearly_rate = yearly_rate[
        (yearly_rate["Year"] >= start_year) & (yearly_rate["Year"] <= end_year)]

    
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(
        yearly_rate["Year"],
        yearly_rate["CurrentInterestRate"],
        color="#2f3b69",      
        marker="o",
        linewidth=2,
        label="Average Current Interest Rate"
    )

    plt.xlabel("Year")
    plt.ylabel("Avg. Current Interest Rate (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    fig_path = Path(fig_dir) / fig_filename
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


    return yearly_rate
