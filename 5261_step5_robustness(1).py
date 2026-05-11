# Sub-period analysis + bootstrap CI

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BT_DIR = "bt_outputs"
OUT_DIR = "robustness_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

RETURNS_PATH = os.path.join(BT_DIR, "daily_returns_all.csv")
SUMMARY_PATH = os.path.join(BT_DIR, "summary_all.csv")

SUBPERIODS = {
    "2016_2022": ("2016-01-01", "2022-12-31"),
    "2023_2025": ("2023-01-01", "2025-10-01"),
}

BOOTSTRAP_B = 1000
RISKFREE = 0.0


def compute_metrics(ret):
    x = pd.Series(ret).dropna()
    if len(x) < 2:
        return {
            "n_obs": len(x),
            "annual_return": np.nan,
            "volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "cumulative_return": np.nan,
        }

    equity = (1 + x).cumprod()
    running_max = equity.cummax()
    drawdown = (equity / running_max - 1.0).min()

    annual_return = (1 + x.mean()) ** 252 - 1
    volatility = x.std(ddof=1) * np.sqrt(252)
    sharpe = np.nan if volatility == 0 else ((x.mean() * 252) - RISKFREE) / volatility
    cumulative_return = equity.iloc[-1] - 1

    return {
        "n_obs": len(x),
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": abs(drawdown) * 100,
        "cumulative_return": cumulative_return,
    }


def bootstrap_sharpe_ci(ret, b=1000, alpha=0.05, seed=42):
    x = pd.Series(ret).dropna().values
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    sharpe_vals = []

    for _ in range(b):
        sample = rng.choice(x, size=n, replace=True)
        vol = np.std(sample, ddof=1) * np.sqrt(252)
        if vol == 0:
            sharpe_vals.append(np.nan)
        else:
            sharpe_vals.append((np.mean(sample) * 252 - RISKFREE) / vol)

    sharpe_vals = pd.Series(sharpe_vals).dropna()
    if len(sharpe_vals) == 0:
        return np.nan, np.nan, np.nan

    lower = sharpe_vals.quantile(alpha / 2)
    median = sharpe_vals.quantile(0.5)
    upper = sharpe_vals.quantile(1 - alpha / 2)
    return lower, median, upper


if __name__ == "__main__":
    if not os.path.exists(RETURNS_PATH):
        raise FileNotFoundError("Missing bt_outputs/daily_returns_all.csv. Run backtest file first.")
    if not os.path.exists(SUMMARY_PATH):
        raise FileNotFoundError("Missing bt_outputs/summary_all.csv. Run backtest file first.")

    returns_df = pd.read_csv(RETURNS_PATH, parse_dates=["date"]).set_index("date")
    summary_df = pd.read_csv(SUMMARY_PATH)

    # Pick the best active strategy by Sharpe ratio
    best_strategy = (
        summary_df[summary_df["strategy"] != "buyhold_equal"]
        .sort_values("sharpe", ascending=False)
        .iloc[0]["strategy"]
    )

    target_strategies = ["buyhold_equal", best_strategy]

    # 1) Sub-period summary
    subperiod_rows = []
    bootstrap_rows = []

    for label, (start, end) in SUBPERIODS.items():
        sub_df = returns_df.loc[start:end, target_strategies]

        for strategy in target_strategies:
            ret = sub_df[strategy].dropna()
            metrics = compute_metrics(ret)

            subperiod_rows.append({
                "period": label,
                "strategy": strategy,
                **metrics
            })

            ci_l, ci_m, ci_u = bootstrap_sharpe_ci(ret, b=BOOTSTRAP_B)
            bootstrap_rows.append({
                "period": label,
                "strategy": strategy,
                "bootstrap_b": BOOTSTRAP_B,
                "sharpe_ci95_lower": ci_l,
                "sharpe_bootstrap_median": ci_m,
                "sharpe_ci95_upper": ci_u,
            })

    subperiod_df = pd.DataFrame(subperiod_rows)
    bootstrap_df = pd.DataFrame(bootstrap_rows)

    subperiod_df.to_csv(os.path.join(OUT_DIR, "robustness_subperiod_summary.csv"), index=False)
    bootstrap_df.to_csv(os.path.join(OUT_DIR, "robustness_bootstrap_ci.csv"), index=False)

    # 2) Plot bootstrap Sharpe CI for benchmark vs best strategy
    plot_df = bootstrap_df.copy()
    plot_df["label"] = plot_df["period"] + " | " + plot_df["strategy"]

    plt.figure(figsize=(8, 4))
    x = np.arange(len(plot_df))
    y = plot_df["sharpe_bootstrap_median"].values
    yerr_lower = y - plot_df["sharpe_ci95_lower"].values
    yerr_upper = plot_df["sharpe_ci95_upper"].values - y

    plt.errorbar(
        x,
        y,
        yerr=[yerr_lower, yerr_upper],
        fmt="o",
        capsize=4
    )
    plt.xticks(x, plot_df["label"], rotation=20, ha="right")
    plt.ylabel("Bootstrap Sharpe Ratio")
    plt.title("Robustness: Bootstrap 95% CI of Sharpe Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "robustness_sharpe_ci.png"), dpi=150)
    plt.close()

    # 3) Save a tiny helper file for report writing
    pd.DataFrame({
        "best_strategy_selected_by_full_sample_sharpe": [best_strategy],
        "benchmark_strategy": ["buyhold_equal"],
        "bootstrap_replications": [BOOTSTRAP_B],
    }).to_csv(os.path.join(OUT_DIR, "robustness_meta.csv"), index=False)

    print("Saved:")
    print("- robustness_outputs/robustness_subperiod_summary.csv")
    print("- robustness_outputs/robustness_bootstrap_ci.csv")
    print("- robustness_outputs/robustness_sharpe_ci.png")
    print("- robustness_outputs/robustness_meta.csv")
