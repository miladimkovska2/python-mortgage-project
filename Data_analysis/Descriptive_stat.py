import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, normaltest
from pathlib import Path


def descriptive_stats_report(
    df: pd.DataFrame,
    col_map: dict,
    *,
    output_dir: str = "Outputs/reports/data_analysis",
    fig_dir: str = "Outputs/Figures/data_analysis",
    filename: str = "descriptive_statistics_report.csv",
    fig_name: str = "distributions_summary.png",
) -> pd.DataFrame:
   
    out = []

    for col, label in col_map.items():
        s = df[col].dropna()
        arr = s.to_numpy()

        # Shape statistics
        sk = skew(arr, bias=False, nan_policy="omit")
        kt = kurtosis(arr, fisher=True, bias=False, nan_policy="omit")

        # Normality test
        if len(arr) >= 8:
            _, p = normaltest(arr)
            normal = "Yes" if p >= 0.05 else "No"
        else:
            p, normal = None, "NA"

        out.append({
            "Variable": label,
            "mean": s.mean(),
            "std": s.std(ddof=1),
            "min": s.min(),
            "25th": s.quantile(0.25),
            "median": s.median(),
            "75th": s.quantile(0.75),
            "max": s.max(),
            "skewness": sk,
            "kurtosis": kt,
            "normality_p": p,
            "normal @ α=0.05": normal
        })

    stats_df = pd.DataFrame(out).set_index("Variable")

    # Save descriptive stats
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(output_path / filename, float_format="%.4f")

    # Plot histograms 
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(15, 4))
    for i, col in enumerate(col_map.keys(), 1):
        plt.subplot(1, len(col_map), i)
        sns.histplot(df[col].dropna(), bins=50, kde=True,
                     color="#2f3b69", edgecolor="black", alpha=0.6)
        plt.title(f"Distribution of {col_map[col]}")
        plt.xlabel(col_map[col])
        plt.ylabel("Density")
        plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    fig_path = fig_dir / fig_name
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()  



    return stats_df
