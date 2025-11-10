import pandas as pd
import numpy as np
from pathlib import Path


working_dir = Path(r"C:\Users\MilaDimkovska\OneDrive - Neon-Advisory\Bureaublad\swap_rates")

# Maturities in years available
maturities = [1, 2, 3, 4, 5, 7, 30]

def file_name_for(term: int) -> str:
    return f"USD {term} Year{'s' if term > 1 else ''} Interest Rate Swap.csv"

# USD fixed leg is typically semiannual 
delta = 0.5

# Output file name
output_filename = "monthly_yield_curve.xlsx"

# Helpers
def pick_alias(col_map_lower_to_orig: dict, aliases: list[str]) -> str | None:
    for a in aliases:
        if a in col_map_lower_to_orig:
            return col_map_lower_to_orig[a]
    return None

def load_and_stack(working_dir: Path, maturities: list[int]) -> pd.DataFrame:
    frames = []
    for term in maturities:
        file_path = working_dir / file_name_for(term)
        try:
            df = pd.read_csv(file_path)

            date_col  = [c for c in df.columns if c.lower() == "observation date"][0]
            price_col = [c for c in df.columns if c.lower() == "price"][0]

            
            df = df[[date_col, price_col]].rename(columns={date_col: "observationdate", price_col: "price"})

            # Clean rate (percent -> decimal)
            df["price"] = (
                df["price"].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False))
            
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["price"] = df["price"] / 100.0  

            # Parse dates
            df["observationdate"] = pd.to_datetime(df["observationdate"], errors="coerce")

            # Attach maturity 
            df["maturity"] = term

            # Drop bad rows
            df = df.dropna(subset=["observationdate", "price"])

            frames.append(df)

        except FileNotFoundError:
            print(f"Warning: File not found - {file_path}")
            continue

    if not frames:
        raise RuntimeError("No swap files were loaded. Check folder path and filenames.")

    return pd.concat(frames, ignore_index=True)


def monthly_collapse(merged_long: pd.DataFrame, how: str = "last") -> pd.DataFrame:
    return merged_long  # no-op: the data is already monthly



def bootstrap_month(month_slice: pd.DataFrame, delta: float = 0.5) -> pd.DataFrame:
    ms = month_slice.sort_values("maturity").reset_index(drop=True).copy()
    ms["df"] = np.nan

    for i in range(len(ms)):
        K = ms.loc[i, "rate"]  
        sum_df = 0.0
        if i > 0:
            sum_df = float((ms.loc[:i-1, "df"] * delta).sum())

        DF_i = (1.0 - K * sum_df) / (1.0 + delta * K)
        ms.loc[i, "df"] = DF_i

    ms["zero_rate"] = -np.log(ms["df"]) / ms["maturity"]
    return ms

def bootstrap_all(monthly_rates: pd.DataFrame, delta: float = 0.5) -> pd.DataFrame:
    out = []
    for month in monthly_rates["monthly"].unique():
        month_data = monthly_rates.loc[monthly_rates["monthly"] == month, ["maturity", "rate"]]
        b = bootstrap_month(month_data, delta=delta)
        b.insert(0, "month", month)
        out.append(b)
    return pd.concat(out, ignore_index=True)




# MAIN
if __name__ == "__main__":
    # Load monthly files and stack
    merged_long = load_and_stack(working_dir, maturities)

    # Data already monthly → build monthly_rates directly
    merged_long["monthly"] = merged_long["observationdate"].dt.to_period("M")
    monthly_rates = merged_long.rename(columns={"price": "rate"})[["monthly", "maturity", "rate"]]

    # Bootstrap
    monthly_yield_curve = bootstrap_all(monthly_rates, delta=delta)

    # Finalize
    monthly_yield_curve = monthly_yield_curve.sort_values(["month", "maturity"]).reset_index(drop=True)
    final_columns = ["month", "maturity", "rate", "df", "zero_rate"]
    monthly_yield_curve = monthly_yield_curve[final_columns]

    # Save
    output_path = working_dir / output_filename
    monthly_yield_curve.to_excel(output_path, index=False)

 
