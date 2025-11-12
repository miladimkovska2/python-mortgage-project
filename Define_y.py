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
    

    # build thresholds use to decide what is a significant prepayment

    # 1. absolute threshold 
    per_loan_installment = (sched.groupby("LoanSequenceNumber")["Monthly Installment"].first())
    median_installment = per_loan_installment.median()
    ABS_EPS = 0.25 * median_installment

    # 2. relative threhold - scale based on the loan size 
    REL_EPS = 0.15 *merged["ScheduledPrincipalCurrent"]

    # Compute the dynamic threshold per observation
    threshold = np.maximum(ABS_EPS, REL_EPS)

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

