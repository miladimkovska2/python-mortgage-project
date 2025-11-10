import numpy as np
import pandas as pd


def build_amortization_schedule(input_path: str) -> pd.DataFrame:
    merged = pd.read_parquet(input_path)
  
    #loan-level parameters 
    loan = (
        merged.sort_values(["LoanSequenceNumber", "MonthlyReportingPeriod"])
         .groupby("LoanSequenceNumber", as_index=False)
         .agg(
             start=("MonthlyReportingPeriod", "first"),
             r_annual=("CurrentInterestRate", "first"),
             maturity=("MaturityDate", "first"),
             UPB=("UPB", "first")
         )
    )

    loan["start"] = pd.to_datetime(loan["start"])
    loan["maturity"] = pd.to_datetime(loan["maturity"])
    loan["r_yr"] = loan["r_annual"] / 100.0
    loan["r_m"] = loan["r_yr"] / 12.0  # monthly rate

    #total number of months till maturity
    loan["n"] = ((loan["maturity"].dt.year - loan["start"].dt.year) * 12 +
                 (loan["maturity"].dt.month - loan["start"].dt.month)).astype(int)

    #Compute the level payment
    loan["A"] = loan["UPB"] * loan["r_m"] / (1.0 - (1.0 + loan["r_m"]) ** (-loan["n"]))


    #Build amortization path
    records = []
    for _, row in loan.iterrows():
        lid = row["LoanSequenceNumber"]
        P0, r, n, A = float(row.UPB), float(row.r_m), int(row.n), float(row.A)
        start_dt = pd.Timestamp(row["start"]).normalize()

        upb_prev = P0
        for k in range(1, n + 1):
            interest_payment = upb_prev * r
            Scheduled_P = A - interest_payment
            upb_k = upb_prev - Scheduled_P

            records.append({
                "LoanSequenceNumber": lid,
                "MonthIndex": k,
                "ContractualUPB": upb_k,
                "Schedueled Interest": interest_payment,
                "Schedueled Principal": Scheduled_P,
                "Monthly Installment": A,
                "ContractualDate": start_dt + pd.DateOffset(months=k)
            })

            upb_prev = upb_k

    schedule = pd.DataFrame(records)
    schedule["ContractualDate"] = pd.to_datetime(schedule["ContractualDate"]).dt.to_period("M").dt.to_timestamp()

    return schedule
