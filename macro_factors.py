import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


base_dir = Path(__file__).resolve().parent
inputs_dir = base_dir / "Inputs"
output_dir = base_dir / "Outputs" / "Figures" / "Data_Analysis"


hpi_path = inputs_dir / "House_Price_Index.csv"
unemp_path = inputs_dir / "unemployment_rate.csv"

# Load data
hpi = pd.read_csv(hpi_path)
unemp = pd.read_csv(unemp_path)

# Parse dates 
hpi["observation_date"] = pd.to_datetime(hpi["observation_date"])
unemp["observation_date"] = pd.to_datetime(unemp["observation_date"])

# Merge both datasets 
df = pd.merge(hpi, unemp, on="observation_date", how="inner")

# Rename for clarity
df.rename(columns={"USSTHPI": "House_Price_Index", "UNRATE": "Unemployment_Rate"}, inplace=True)


fig, ax1 = plt.subplots(figsize=(10, 5))

# Left y-axis: House Price Index
ax1.plot(df["observation_date"], df["House_Price_Index"], color="#2f3b69", label="House Price Index")
ax1.set_xlabel("Date")
ax1.set_ylabel("House Price Index", color="#2f3b69")
ax1.tick_params(axis="y", labelcolor="#2f3b69")

# Right y-axis: Unemployment Rate
ax2 = ax1.twinx()
ax2.plot(df["observation_date"], df["Unemployment_Rate"], color="#c197d2", label="Unemployment Rate")
ax2.set_ylabel("Unemployment Rate (%)", color="#c197d2")
ax2.tick_params(axis="y", labelcolor="#c197d2")

# Title & layout
plt.title("House Price Index and Unemployment Rate Over Time")
plt.grid(alpha=0.3)
plt.tight_layout()

# Save plot 
save_path = output_dir / "HPI_and_Unemployment.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close(fig)

