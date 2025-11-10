# plot_upb.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_upb_actual_vs_contractual(
    merged_path: str,               
    amort_schedule_path: str,      
    *,
    start_year: int = 2010,
    end_year: int = 2025,
    fig_dir: str = "Outputs/Figures/data_analysis",
    fig_filename: str = "upb_actual_vs_contractual.png",
) -> pd.DataFrame:
   
    # Load inputs 
    merged = pd.read_parquet(merged_path)
    schedule = pd.read_parquet(amort_schedule_path)

    # Actual UPB by year 
    merged = merged.copy()
    merged["MonthlyReportingPeriod"] = pd.to_datetime(merged["MonthlyReportingPeriod"])
    merged["Year"] = merged["MonthlyReportingPeriod"].dt.year

    actual_by_year = (
        merged.loc[(merged["Year"] >= start_year) & (merged["Year"] <= end_year)]
              .groupby("Year", as_index=False)["CurrentActualUPB"]
              .mean()
              .rename(columns={"CurrentActualUPB": "ActualUPB"})
    )

    # Contractual UPB by year (from prebuilt amortization schedule) 
    schedule = schedule.copy()
    schedule["ContractualDate"] = pd.to_datetime(schedule["ContractualDate"])
    schedule["Year"] = schedule["ContractualDate"].dt.year

    contractual_by_year = (
        schedule.loc[(schedule["Year"] >= start_year) & (schedule["Year"] <= end_year)]
                .groupby("Year", as_index=False)["ContractualUPB"]
                .mean()
    )

    combined = (actual_by_year
                .merge(contractual_by_year, on="Year", how="outer")
                .sort_values("Year")
                .reset_index(drop=True))

    # Plot 
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(combined["Year"], combined["ActualUPB"], marker="o", linestyle="-", color="#2f3b69", label="Actual average UPB")
    plt.plot(combined["Year"], combined["ContractualUPB"], marker="o", linestyle="--", color="#c197d2", label="Contractual average UPB")
    plt.title(f"Average UPB – Actual vs Contractual ({start_year}–{end_year})")
    plt.xlabel("Year")
    plt.ylabel("Average UPB")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    fig_path = Path(fig_dir) / fig_filename
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    return combined

