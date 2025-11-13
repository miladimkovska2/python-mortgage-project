
import pandas as pd
from pathlib import Path
from src.format_variables_mortgages import format_datasets

def load_freddie_mac_data(input_dir, output_dir="Outputs"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Parquet file paths
    orig_parquet = output_dir / "orig_formatted.parquet"
    perf_parquet = output_dir / "perf_formatted.parquet"

    '''
    #If formatted Parquet files exist, load them directly
    if orig_parquet.exists() and perf_parquet.exists():
        orig = pd.read_parquet(orig_parquet)
        perf = pd.read_parquet(perf_parquet)
        return orig, perf
    '''
    #read the raw text files
    orig_file = input_dir / "sample_orig_2010.txt"
    perf_file = input_dir / "sample_svcg_2010.txt"

    # Define column names 
    orig_cols = [
        "CreditScore","FirstPaymentDate","FirstTimeHomebuyerFlag","MaturityDate",
        "MSA","MI_Percent","NumberOfUnits","OccupancyStatus","CLTV","DTI",
        "UPB","LTV","InterestRate","Channel","PPM_Flag","AmortizationType",
        "PropertyState","PropertyType","PostalCode","LoanSequenceNumber",
        "LoanPurpose","LoanTerm","NumBorrowers","SellerName","ServicerName",
        "SuperConformingFlag","PreHARP_SequenceNumber","ProgramIndicator",
        "HARP_Indicator","PropertyValuationMethod","InterestOnlyFlag",
        "MICancelIndicator"
    ]

    perf_cols = [
        "LoanSequenceNumber","MonthlyReportingPeriod","CurrentActualUPB",
        "CurrentLoanDelinquencyStatus","LoanAge","MonthsToMaturity","DefectSettlementDate",
        "ModificationFlag","ZeroBalanceCode","ZeroBalanceEffectiveDate",
        "CurrentInterestRate","CurrentDeferredUPB","DDLPI","MIRecoveries",
        "NetSalesProceeds","NonMIRecoveries","Expenses","LegalCosts",
        "MaintenanceCosts","TaxesInsurance","MiscExpenses","ActualLossCalculation",
        "ModificationCost","StepModificationFlag","DeferredPaymentPlan",
        "EstimatedLTV","ZeroBalanceRemovalUPB","DelinquentAccruedInterest",
        "DelinquencyDueToDisaster","BorrowerAssistanceStatusCode",
        "CurrentMonthModificationCost","InterestBearingUPB"
    ]

    orig = pd.read_csv(orig_file, sep="|", header=None, names=orig_cols, low_memory=False)
    perf = pd.read_csv(perf_file, sep="|", header=None, names=perf_cols, low_memory=False)

    # Select relevant variables
    orig = orig[["LoanSequenceNumber", "PPM_Flag", "MaturityDate", "InterestOnlyFlag", "UPB", "PropertyState","PropertyType"]]
    perf = perf[["LoanSequenceNumber", "CurrentActualUPB", "MonthlyReportingPeriod",
        "ZeroBalanceCode", "ZeroBalanceEffectiveDate", "CurrentInterestRate",
        "EstimatedLTV", "ModificationFlag", "LoanAge"]]

    # Format variables (types, categories, dates)
    orig, perf = format_datasets(orig, perf)

    # Save formatted data as Parquet
    orig.to_parquet(orig_parquet, index=False)
    perf.to_parquet(perf_parquet, index=False)

    return orig, perf
