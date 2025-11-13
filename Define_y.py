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
    

    # build thresholds use to decide what is a significant prepayment - subjective choice 
    per_loan_installment = (sched.groupby("LoanSequenceNumber")["Monthly Installment"].first())

    # Absolute epsilon per loan: 15% of that installment
    ABS_EPS_by_loan = 0.15 * per_loan_installment

    # Map this per-loan threshold to each row in merged
    ABS_EPS_per_row = merged["LoanSequenceNumber"].map(ABS_EPS_by_loan)

    #minimum in dollars - since for a very small loans such as of 10K UPB the abs_eps will be also very small
    MIN_EPS = 200.0

    threshold = ABS_EPS_per_row.clip(lower=MIN_EPS)

    
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

