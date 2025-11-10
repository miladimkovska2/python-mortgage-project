import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def interest_loss_from_schedule(
    merged_path: str,
    amort_schedule_path: str,
    *,
    plot: bool = True,
    fig_dir: str = "Outputs/Figures/data_analysis",
    fig_filename: str = "cumulative_interest_loss.png",
):
    # Load & normalize dates
    merged = pd.read_parquet(merged_path).copy()
    sched  = pd.read_parquet(amort_schedule_path).copy()
    merged["MonthlyReportingPeriod"] = pd.to_datetime(merged["MonthlyReportingPeriod"]).dt.to_period("M").dt.to_timestamp()
    sched["ContractualDate"]         = pd.to_datetime(sched["ContractualDate"]).dt.to_period("M").dt.to_timestamp()

    last_actual = merged["MonthlyReportingPeriod"].max()
    

    # Fixed monthly rate per loan 
    base = (merged.sort_values(["LoanSequenceNumber","MonthlyReportingPeriod"])
                  .groupby("LoanSequenceNumber", as_index=False)
                  .agg(r_annual=("CurrentInterestRate","first")))
    base["r_m"] = (base["r_annual"].astype(float) / 100.0) / 12.0


    # Build the full schedule panel (baseline)
    panel = (sched[["LoanSequenceNumber","ContractualDate","Schedueled Interest"]]
                  .rename(columns={"ContractualDate":"Period"})
                  .merge(base[["LoanSequenceNumber","r_m"]], on="LoanSequenceNumber", how="left"))
    

    # previous month's actual UPB 
    merged = merged.sort_values(["LoanSequenceNumber","MonthlyReportingPeriod"])
    merged["Act_UPB_prev"] = merged.groupby("LoanSequenceNumber")["CurrentActualUPB"].shift(1)
    act_period = merged[["LoanSequenceNumber","MonthlyReportingPeriod","Act_UPB_prev"]] \
                      .rename(columns={"MonthlyReportingPeriod":"Period"})


    # after full prepay, Act_UPB_prev stays NaN -> set to 0
    detail = (panel.merge(act_period, on=["LoanSequenceNumber","Period"], how="left"))
    detail["Act_UPB_prev"] = detail["Act_UPB_prev"].fillna(0.0)

    detail = detail[ detail["Period"] <= last_actual ]

    # Interest loss
    detail["Interest_Actual"]    = detail["r_m"] * detail["Act_UPB_prev"]
    detail["Interest_Scheduled"] = detail["Schedueled Interest"]
    detail["Interest_Loss"]      = detail["Interest_Scheduled"] - detail["Interest_Actual"]

    # cumulative
    portfolio = (detail.groupby("Period", as_index=False)["Interest_Loss"]
                       .sum()
                       .sort_values("Period"))
    portfolio["Cum_Int_Loss"] = portfolio["Interest_Loss"].cumsum()


    detail["Year"] = pd.to_datetime(detail["Period"]).dt.year
    annual = (detail.groupby("Year", as_index=False)["Interest_Loss"]
                    .sum()
                    .rename(columns={"Interest_Loss":"Interest_Loss_Year"}))
    annual["Cum_Int_Loss_Year"] = annual["Interest_Loss_Year"].cumsum()

    # Plot 
    fig_path = None
    if plot:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(12,6))
        plt.plot(portfolio["Period"], portfolio["Cum_Int_Loss"], linewidth=2.2, marker="o", markersize=3,
                 label="Cumulative Interest Loss")
        plt.xlabel("Date"); plt.ylabel("€"); plt.title("Cumulative Interest Loss")
        plt.grid(True, linestyle="--", alpha=0.6); plt.legend(); plt.tight_layout()
        fig_path = Path(fig_dir) / fig_filename
        plt.savefig(fig_path, dpi=300, bbox_inches="tight"); plt.close()

    return detail, portfolio, fig_path