import pandas as pd
import numpy as np


def format_datasets(orig, perf):

    # origination format
    orig["LoanSequenceNumber"] = orig["LoanSequenceNumber"].astype(str)
    orig["MaturityDate"] = pd.to_datetime(orig["MaturityDate"], format="%Y%m", errors="coerce")

    orig["PPM_Flag"] = (orig["PPM_Flag"].astype(str).str.strip().replace({"Y": 1, "N": 0, "": np.nan}).astype("Int64"))
    orig["PPM_Flag"] = pd.to_numeric(orig["PPM_Flag"], errors="coerce").astype("Int64")

# InterestOnlyFlag: 'Y' = 1, 'N' = 0
    orig["InterestOnlyFlag"] = (orig["InterestOnlyFlag"].astype(str).str.strip().replace({"Y": 1, "N": 0, "": np.nan}).astype("Int64"))
    orig["InterestOnlyFlag"] = pd.to_numeric(orig["InterestOnlyFlag"], errors="coerce").astype("Int64")

    orig["UPB"] = pd.to_numeric(orig["UPB"], errors="coerce")

    # performance formt
    perf["LoanSequenceNumber"] = perf["LoanSequenceNumber"].astype(str)
    perf["MonthlyReportingPeriod"] = pd.to_datetime(perf["MonthlyReportingPeriod"], format="%Y%m", errors="coerce")
    perf["CurrentActualUPB"] = pd.to_numeric(perf["CurrentActualUPB"], errors="coerce")
    perf["CurrentInterestRate"] = pd.to_numeric(perf["CurrentInterestRate"], errors="coerce")
    perf["EstimatedLTV"] = pd.to_numeric(perf["EstimatedLTV"], errors="coerce")
    perf["EstimatedLTV"] = perf["EstimatedLTV"].replace({999: np.nan})
    perf["ZeroBalanceEffectiveDate"] = pd.to_datetime(perf["ZeroBalanceEffectiveDate"],format="%Y%m",errors="coerce")

    # categorical var 
    # ZeroBalanceCode: blank = no event occurs
    perf["ZeroBalanceCode"] = (
    perf["ZeroBalanceCode"]
    .replace(" ", np.nan)
    .fillna("not_applicable")   
    .astype(str)                
)                               
    perf["ZeroBalanceCode"] = perf["ZeroBalanceCode"].astype("category")

# ModificationFlag: blank = not modified
    perf["ModificationFlag"] = (
    perf["ModificationFlag"]
    .replace(" ", np.nan)
    .fillna("not_modified")
    .astype(str)
)
    perf["ModificationFlag"] = perf["ModificationFlag"].astype("category")


    return orig, perf
