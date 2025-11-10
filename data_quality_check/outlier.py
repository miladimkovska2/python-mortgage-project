import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path



# Custom color palette
BLUE, GREY, PURPLE = "#2f3b69", "#9f9f9f", "#c197d2"


def _safe_std(x: pd.Series) -> float:
    return float(np.nanstd(x.to_numpy(), ddof=1)) if x.size > 1 else 0.0

def _mad(x: pd.Series) -> float:
    x = x.dropna()
    if x.empty:
        return np.nan
    med = np.median(x)
    return np.median(np.abs(x - med))


def _modified_z_scores(x: pd.Series) -> np.ndarray:
    arr = x.to_numpy(dtype=float)
    mask = ~np.isnan(arr)
    out = np.full_like(arr, np.nan)
    if mask.sum() == 0:
        return out
    med = np.nanmedian(arr)
    mad = _mad(pd.Series(arr[mask]))
    if mad == 0 or np.isnan(mad):
        out[mask] = 0.0
        return out
    out[mask] = 0.6745 * (arr[mask] - med) / mad
    return out


def outlier_report(
    df: pd.DataFrame,
    cols: list[str],
    z_thr: float = 3.0,
    mz_thr: float = 3.5,
    output_dir: str = "Outputs/reports/Quality_Results",   
    filename: str = "outlier_report.png"
) -> pd.DataFrame:
    
    rows = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        n = s.notna().sum()
        if n == 0:
            rows.append({
                "column": c,
                "IQR_outliers_%": np.nan,
                "Z_outliers_%": np.nan,
                "MZ_outliers_%": np.nan,
            })
            continue

        # IQR
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_mask = (s < lower) | (s > upper)
        iqr_pct = 100.0 * iqr_mask.sum() / n

        # Classical Z 
        mu, sd = s.mean(), _safe_std(s)
        z_pct = 100.0 * ((np.abs((s - mu) / sd) > z_thr).sum() / n) if sd > 0 else 0.0

        # Modified Z 
        mz = _modified_z_scores(s)
        mz_pct = 100.0 * np.nansum(np.abs(mz) > mz_thr) / n

        rows.append({
            "column": c,
            "Q1": q1, "Q3": q3, "IQR": iqr,
            "IQR_lower": lower, "IQR_upper": upper,
            "IQR_outliers_%": iqr_pct,
            "Z_outliers_%": z_pct,
            "MZ_outliers_%": mz_pct,
        })

        # Boxplots
        fig, ax = plt.subplots(figsize=(6, 4))
        box = ax.boxplot(s.dropna(), patch_artist=True,
                         boxprops=dict(facecolor=PURPLE, color=BLUE),
                         whiskerprops=dict(color=GREY),
                         capprops=dict(color=GREY),
                         medianprops=dict(color=BLUE, linewidth=2),
                         flierprops=dict(marker='o', color=BLUE, alpha=0.4))
        ax.set_title(f"Boxplot of {c}", fontsize=12, color=BLUE)
        ax.set_ylabel(c, color=GREY)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / f"boxplot_{c}.png", dpi=300, bbox_inches="tight")
        plt.close()


    # Summary Table
    report = pd.DataFrame(rows).set_index("column")

    fig, ax = plt.subplots(figsize=(10, len(report) * 0.5 + 1))
    ax.axis("off")
    ax.table(
        cellText=np.round(report.values, 3),
        colLabels=report.columns,
        rowLabels=report.index,
        cellLoc="center",
        loc="center",
    )
    plt.title("Outlier Detection Summary", fontsize=12, pad=10, color=BLUE)
    plt.tight_layout()
    plt.savefig(output_path / filename, dpi=300, bbox_inches="tight")
    plt.close()

   
    # Compute score 
    valid_iqr = report["IQR_outliers_%"].dropna()
    sum_iqr = valid_iqr.sum() if not valid_iqr.empty else np.nan
    outlier_score = round(1 - sum_iqr / 100, 3) if pd.notna(sum_iqr) else np.nan


    return report, outlier_score
