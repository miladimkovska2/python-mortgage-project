import pandas as pd
import numpy as np

def add_prepayment_flags(merged: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    
    merged = merged.copy()
    sched = sched.copy()


    merged["MonthlyReportingPeriod"] = pd.to_datetime(merged["MonthlyReportingPeriod"]).dt.to_period("M").dt.to_timestamp()

    sched["ContractualDate"] = pd.to_datetime(sched["ContractualDate"]).dt.to_period("M").dt.to_timestamp()


    # Align scheduled principal
    sched = sched.rename(columns={"ContractualDate": "MonthlyReportingPeriod",
                                  "Schedueled Principal": "ScheduledPrincipalCurrent"})


    merged = merged.merge(
        sched[["LoanSequenceNumber", "MonthlyReportingPeriod", "ScheduledPrincipalCurrent"]],
        on=["LoanSequenceNumber", "MonthlyReportingPeriod"], how="left")


    # Previous UPB per loan
    merged = merged.sort_values(["LoanSequenceNumber","MonthlyReportingPeriod"])
    merged["UPB_prev"] = merged.groupby("LoanSequenceNumber")["CurrentActualUPB"].shift(1)


    # Actual principal collected 
    merged["ActualPrincipalCollected"] = (
        (merged["UPB_prev"] - merged["CurrentActualUPB"]).astype(float)).fillna(0.0)  


    # Compute partial prepayment amount 
    merged["PartialPrepayAmt"] = (
        merged["ActualPrincipalCollected"].round(2) - merged["ScheduledPrincipalCurrent"].round(2))
    
    
    ABS_EPS = 0.25 * 1212.77      # 25% of average installment (= $1, 212.77) ≈ $303
    REL_EPS = 0.15                # 15% of scheduled principal

    # Compute the dynamic threshold per observation
    threshold = np.maximum(ABS_EPS, REL_EPS * merged["ScheduledPrincipalCurrent"])

# Zero out anything smaller than both the absolute and relative threshold
    merged["PartialPrepayAmt"] = np.where(merged["PartialPrepayAmt"] > threshold, merged["PartialPrepayAmt"],0.0)


    # Indicators 
    partial = ((merged["PartialPrepayAmt"] > 0) & (merged["ZeroBalanceCode"] == "not_applicable"))

    full = (merged["ZeroBalanceCode"] == "1.0")

    # Categorical column
    merged["PrepayType"] = 0
    merged.loc[full, "PrepayType"] = 1
    merged.loc[partial, "PrepayType"] = 2

    merged["PrepayType"] = merged["PrepayType"].astype(int)

    return merged

