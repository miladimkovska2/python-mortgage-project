import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
import os
import matplotlib.pyplot as plt
import seaborn as sns


# Load Mortgage Data and format variables
from src.load_data_mortgages import load_freddie_mac_data

orig, perf = load_freddie_mac_data(Path("Inputs"), Path("Outputs"))


# Data Quality Framework

# 1.Data Accuracy and Validity
from data_quality_check.accuracy_validity import run_accuracy_validity_score


valid_zbc = {'1.0', '2.0', '3.0', '6.0', '9.0', '15.0', '16.0', '96.0', "not_applicable"}

rules = {
        "perf": [
            {"col": "CurrentInterestRate", "condition": lambda x: x <= 0},
            {"col": "CurrentActualUPB", "condition": lambda x: x < 0},
            {"col": "EstimatedLTV", "condition": lambda x: x < 0},
            {"col": "ZeroBalanceCode", "condition": lambda x: ~x.isin(valid_zbc)},
            {"col": "CurrentInterestRate", "condition": lambda x: ~x.apply(lambda v: isinstance(v, (int, float)))},
            {"col": "LoanAge", "condition": lambda x: ~x.apply(lambda v: isinstance(v, (int, float)))},
            {"col": "CurrentActualUPB", "condition": lambda x: ~x.apply(lambda v: isinstance(v, (int, float)))},
            {"col": "EstimatedLTV", "condition": lambda x: ~x.apply(lambda v: isinstance(v, (int, float)))},
        ],
        "orig": [
            {"col": "UPB", "condition": lambda x: x < 0},
            {"col": "UPB", "condition": lambda x: ~x.apply(lambda v: isinstance(v, (int, float)))},
            {"col": "PPM_Flag", "condition": lambda x: ~x.isin([0, 1])},
            {"col": "InterestOnlyFlag", "condition": lambda x: ~x.isin([0, 1])},
            {"col": "PropertyState", "condition": lambda x: x.str.len() != 2},
            {"col": "PropertyType","condition": lambda x: ~x.isin(["SF", "CO", "PU", "MH", "CP"])},
        ]
    }

df_dict = {"orig": orig, "perf": perf}
dq_scores = run_accuracy_validity_score(df_dict, rules)


# 2. Data Completeness
from data_quality_check.completeness import completeness_score

complet_score, orig, perf = completeness_score(
    df1=orig,
    df2=perf,
    id_col="LoanSequenceNumber",
    date_col="MonthlyReportingPeriod",
    first_period_after_year=2011, 
    exclude_cols=["ZeroBalanceEffectiveDate"])



# 3. Data Consistency 
from data_quality_check.consistency import run_consistency_checks

orig, perf, results = run_consistency_checks(
    df1=orig,
    df2=perf,
    id_col="LoanSequenceNumber",
    date_col="MonthlyReportingPeriod",
    cross_field_tuple=("ZeroBalanceEffectiveDate", "ZeroBalanceCode", "CurrentActualUPB"),
    rate_col="CurrentInterestRate",
    mod_col="ModificationFlag",
    output_dir="Outputs/reports/Quality_Results")


# 4. Data Uniqueness
from data_quality_check.uniqueness import uniqueness_score

uniq_score = uniqueness_score(
    df1=orig,
    df2=perf,
    id_cols_df1=["LoanSequenceNumber"],
    id_cols_df2=["LoanSequenceNumber", "MonthlyReportingPeriod"])




# 5. Data Outliers
from data_quality_check.outlier import outlier_report

numeric_cols = ["CurrentInterestRate", "EstimatedLTV", "CurrentActualUPB"]

report, outlier_score = outlier_report(
    df=perf,
    cols=numeric_cols,
    filename="outlier_report_perf.png")


# 6. Data Representativeness

from data_quality_check.representativeness import (check_representativeness, compute_overall_representativeness_score)

# Run for performance dataset
tables_perf = check_representativeness(
    df=perf,
    categorical_cols=["ZeroBalanceCode"],
    numeric_cols=["CurrentActualUPB", "CurrentInterestRate", "EstimatedLTV"],
    output_dir="Outputs/reports/Quality_Results",
    image_name="representativeness_perf.png"
)

# Run for origination dataset
tables_orig = check_representativeness(
    df=orig,
    binary_cols=["PPM_Flag", "InterestOnlyFlag"],
    numeric_cols=["UPB"],
    categorical_cols=["PropertyState", "PropertyType"],
    output_dir="Outputs/reports/Quality_Results",
    image_name="representativeness_orig.png")

rep_score = compute_overall_representativeness_score(tables_perf, tables_orig)


# 7. Data Quality Summary Table

accuracy_score = dq_scores      
completeness_score_val = complet_score   
consistency_score_val = results["Consistency_Score"]
uniqueness_score_val = uniq_score     
representativeness_score_val = rep_score 
outlier_score_val = outlier_score

# Create summary table
summary_df = pd.DataFrame({
    "Data Quality Dimension": [
        "Accuracy & Validity",
        "Completeness",
        "Consistency",
        "Uniqueness",
        "Outliers",
        "Representativeness"
    ],
    "Score": [
        round(accuracy_score, 3),
        round(completeness_score_val, 3),
        round(consistency_score_val, 3),
        round(uniqueness_score_val, 3),
        np.nan if pd.isna(outlier_score_val) else round(outlier_score_val, 3),
        round(representativeness_score_val, 3)
    ]
})

# Save 
output_path = Path("Outputs/reports/Quality_Results")
output_path.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(output_path / "data_quality_summary.csv", index=False)


############################################################################################################

# Merge origination + performance based on LoanSequenceNumber
merged = perf.merge(orig, on="LoanSequenceNumber", how="left")
merged.to_csv("Outputs/merged.csv", index=False)

# Convert date-like columns 
date_cols = ["MonthlyReportingPeriod", "ZeroBalanceEffectiveDate", "MaturityDate"]
for c in date_cols:
    if c in merged.columns:
        merged[c] = (
            merged[c]
            .astype(str)
            .str.strip()
            .replace({"": None, "NaN": None, "nan": None})
        )
        merged[c] = pd.to_datetime(merged[c], errors="coerce")



############################################################################################################


# Data Analysis

#Table 2
from Data_analysis.contractual import loan_summary_report
loan_summary_report(merged)



# Table 3 
from Data_analysis.maturity import maturity_summary_report
maturity_summary = maturity_summary_report(df=merged, cutoff_year=2025)



# Plot active vs. contractual active loans
from Data_analysis.active_vs_contractual import plot_active_vs_contractual_loans
combined = plot_active_vs_contractual_loans(merged)



# Table 4
from Data_analysis.count_zero_balance_code import loan_termination_report

terminated_loans = loan_termination_report(
    df=merged,
    id_col="LoanSequenceNumber",
    date_col="MonthlyReportingPeriod",
    termination_col="ZeroBalanceCode")



#Table 5 - Descriptive stat.
from Data_analysis.Descriptive_stat import descriptive_stats_report

cols = { 
    "CurrentActualUPB": "Current Actual UPB",
    "CurrentInterestRate": "Current Interest Rate (%)",
    "EstimatedLTV": "Estimated LTV",
}

desc_stats = descriptive_stats_report(merged, cols)




# Build the contractual amortization plan
from Data_analysis.contractual_path import build_amortization_schedule

input_path = "Outputs/merged.csv"      
output_path = "Outputs/amortization_schedule.parquet"

amort_schedule = build_amortization_schedule(input_path)
amort_schedule.to_parquet(output_path, index=False)


output_path = "Outputs/amortization_schedule.parquet"


# Plot actual UPB vs contractual
from Data_analysis.actual_vs_contractual_UPB import plot_upb_actual_vs_contractual

merged_path = "Outputs/merged.csv"
amort_schedule_path = "Outputs/amortization_schedule.parquet"

combined = plot_upb_actual_vs_contractual(
    merged_path,
    amort_schedule_path,
    start_year=2010,
    end_year=2025,
    fig_dir="Outputs/Figures/data_analysis",
    fig_filename="upb_actual_vs_contractual.png")


# Loss due to early exit
from Data_analysis.interest_loss_income import interest_loss_from_schedule

detail, portfolio, fig_path = interest_loss_from_schedule(
    merged_path="Outputs/merged.csv",
    amort_schedule_path="Outputs/amortization_schedule.parquet",
    plot=True,
    fig_dir="Outputs/Figures/data_analysis",
    fig_filename="cumulative_interest_loss.png")



# LTV over time
from Data_analysis.plot_LTV import plot_estimated_ltv_trend
ltv_trend = plot_estimated_ltv_trend(merged)



# Remaining avg. interest rate 
from Data_analysis.plot_interest import plot_interest_rate_trend
rate_trend = plot_interest_rate_trend(merged)



# Define Dependent variable                       
from Define_y import add_prepayment_flags

merged = pd.read_csv("Outputs/merged.csv")
sched  = pd.read_csv("Outputs/amortization_schedule.csv")

merged_flags = add_prepayment_flags(merged, sched)

dummy_df = merged_flags[["LoanSequenceNumber", "MonthlyReportingPeriod", "PrepayType"]].copy()

for df in (merged, dummy_df):
    df["MonthlyReportingPeriod"] = pd.to_datetime(df["MonthlyReportingPeriod"]).dt.to_period("M").dt.to_timestamp()

merged = merged.merge(
    dummy_df,
    on=["LoanSequenceNumber", "MonthlyReportingPeriod"],
    how="left")

merged.to_csv("Outputs/merged.csv", index=False)



############ PLots from Bivariate  ############################################

#Corr matrix
from Data_analysis.corr import plot_correlation_matrix
corr_matrix = plot_correlation_matrix(merged)


## PIE 
BLUE   = "#2f3b69"
GREY   = "#9f9f9f"
PURPLE = "#c197d2"
palette = [BLUE, GREY, PURPLE, "#b7b7b7", "#d9c7e8"]  


save_path = "Outputs/figures/Data_analysis/"
os.makedirs(save_path, exist_ok=True)

# Count unique loans per state
state_counts = (
    merged.groupby("PropertyState")["LoanSequenceNumber"]
      .nunique()
      .sort_values(ascending=False))

# Use top 10 states
top10 = state_counts.head(10)

# Define color palette for slices
palette = [BLUE, GREY, PURPLE] + [
    plt.cm.Blues(0.4),
    plt.cm.Greys(0.5),
    plt.cm.Purples(0.5),
    plt.cm.Blues(0.6),
    plt.cm.Greys(0.7),
    plt.cm.Purples(0.7),]

# Plot
plt.figure(figsize=(8, 8))
plt.pie(
    top10.values,
    labels=top10.index,
    autopct='%1.1f%%',
    colors=palette[:len(top10)]
)
plt.title("Top 10 Property States by Unique Mortgages", color=BLUE)
plt.tight_layout()

# Save figure 
plt.savefig(save_path + "property_state_distribution.png", dpi=300)
plt.close()  



# PLOT per property
save_path = "Outputs/figures/Data_analysis/"
os.makedirs(save_path, exist_ok=True)

# Count unique loans per property type
ptype_counts = (
    merged.groupby("PropertyType")["LoanSequenceNumber"]
      .nunique()
      .sort_values(ascending=False))

plt.figure(figsize=(8, 5))
plt.bar(ptype_counts.index, ptype_counts.values, color=palette[:len(ptype_counts)])

plt.xlabel("Property Type", color=BLUE)
plt.ylabel("Number of Unique Loans", color=BLUE)
plt.title("Distribution of Unique Mortgages by Property Type", color=BLUE)
plt.tight_layout()

# Save figure
plt.savefig(save_path + "property_type_distribution.png", dpi=300)
plt.close()



# Boxplot Full Prepayment
save_path = "Outputs/figures/Data_analysis/"
os.makedirs(save_path, exist_ok=True)

# Filter full prepayments only (PrepayType = 1)
full_prepay = merged[merged["PrepayType"] == 1]

plt.figure(figsize=(8, 6))
plt.boxplot(full_prepay["LoanAge"], vert=True,
            patch_artist=True,
            boxprops=dict(facecolor=BLUE, color="black"),
            medianprops=dict(color="white", linewidth=2))

plt.title("Distribution of Loan Age at Full Prepayment", color=BLUE)
plt.ylabel("Loan Age at Full Prepayment (months)", color=BLUE)
plt.tight_layout()

plt.savefig(save_path + "loan_age_full_prepayment.png", dpi=300)
plt.close()   



# Boxplot Partial Prepayment
save_path = "Outputs/figures/Data_analysis/"
os.makedirs(save_path, exist_ok=True)

partial_prepay = merged[merged["PrepayType"] == 2]

plt.figure(figsize=(8, 6))
plt.boxplot(
    partial_prepay["LoanAge"],
    vert=True,
    patch_artist=True,
    boxprops=dict(facecolor=BLUE, color="black"),
    medianprops=dict(color="white", linewidth=2)
)

plt.title("Distribution of Loan Age at Partial Prepayment", color=BLUE)
plt.ylabel("Loan Age at Partial Prepayment (months)", color=BLUE)
plt.tight_layout()

plt.savefig(save_path + "loan_age_partial_prepayment.png", dpi=300)
plt.close()


# Seasonality check - do we see some pronouce effect of prepayment in some months

merged = pd.read_csv("Outputs/merged.csv")
save_path = "Outputs/figures/Data_analysis/"
os.makedirs(save_path, exist_ok=True) 

merged["Month"] = pd.to_datetime(merged["MonthlyReportingPeriod"]).dt.month

all_counts = (merged[merged["PrepayType"].isin([1,2])].groupby("Month").size())

seasonality_pct = (all_counts / all_counts.sum()) *100

plt.figure(figsize=(10,5))
plt.plot(seasonality_pct.index, seasonality_pct.values, marker='o', color=BLUE)
plt.xlabel("Month")
plt.ylabel("Percentage of Yearly Prepayments")
plt.title("Monthly Prepayment Seasonality (Normalized)")
plt.grid(True)
plt.xticks(range(1,13))
plt.savefig(save_path + "seasonality.png", dpi=300)
plt.close()


