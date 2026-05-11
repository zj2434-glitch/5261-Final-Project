import os
import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib.pyplot as plt

 
DATA_DIR = "data_with_signals"
OUT_DIR = "bt_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

START = "2016-01-01"
END = "2025-10-01"

CASH = 100000.0
COMMISSION = 0.001
RISKFREE = 0.0


class SignalData(bt.feeds.PandasData):
    lines = ("signal",)
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", None),
        ("signal", "signal"),
    )


class SignalStrategy(bt.Strategy):
    params = dict(target_pct=0.2)

    def next(self):
        for d in self.datas:
            sig = d.signal[0]
            pos = self.getposition(d).size

            if sig > 0 and pos == 0:
                self.order_target_percent(data=d, target=self.p.target_pct)

            elif sig <= 0 and pos != 0:
                self.order_target_percent(data=d, target=0.0)


class BuyAndHold(bt.Strategy):
    params = dict(target_pct=0.2)

    def __init__(self):
        self.invested = False

    def next(self):
        if not self.invested:
            for d in self.datas:
                self.order_target_percent(data=d, target=self.p.target_pct)
            self.invested = True


def load_data(path, signal_col):
    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError(f"{path} missing date column")

    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= START) & (df["date"] <= END)]
    df = df.sort_values("date")

    required_cols = ["open", "high", "low", "close", "volume"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"{path} missing {c} column")

    if signal_col not in df.columns:
        raise ValueError(f"{path} missing {signal_col}")

    signal = df[signal_col].fillna(0).astype(float)

    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["signal"] = signal.values
    df.set_index("date", inplace=True)

    return df


def run_backtest(signal_col, tag, is_benchmark=False):
    cerebro = bt.Cerebro(stdstats=False)

    csvs = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".csv")
    ]

    if not csvs:
        raise FileNotFoundError("No signal files found. Run Step 2 first.")

    for path in csvs:
        symbol = os.path.basename(path).split("_")[0]
        df = load_data(path, signal_col)
        cerebro.adddata(SignalData(dataname=df), name=symbol)

    n = len(cerebro.datas)
    target_pct = 1.0 / n

    cerebro.broker.setcash(CASH)
    cerebro.broker.setcommission(commission=COMMISSION)

    if is_benchmark:
        cerebro.addstrategy(BuyAndHold, target_pct=target_pct)
    else:
        cerebro.addstrategy(SignalStrategy, target_pct=target_pct)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, riskfreerate=RISKFREE, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.Returns, tann=252, _name="rets")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timeret")

    result = cerebro.run()
    strat = result[0]

    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio", np.nan)
    dd = strat.analyzers.dd.get_analysis()
    rets = strat.analyzers.rets.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    daily_ret = pd.Series(strat.analyzers.timeret.get_analysis())
    daily_ret.index = pd.to_datetime(daily_ret.index)
    daily_ret = daily_ret.fillna(0)

    equity = (1 + daily_ret).cumprod()

    summary = {
        "strategy": tag,
        "signal_col": signal_col,
        "final_value": cerebro.broker.getvalue(),
        "total_return": rets.get("rtot"),
        "annual_return": rets.get("rnorm"),
        "volatility": daily_ret.std() * np.sqrt(252),
        "max_drawdown": dd.get("max", {}).get("drawdown"),
        "sharpe": sharpe,
        "total_trades": trades.get("total", {}).get("total"),
        "won": trades.get("won", {}).get("total"),
        "lost": trades.get("lost", {}).get("total"),
    }

    pd.DataFrame({
        "date": equity.index,
        "equity": equity.values
    }).to_csv(os.path.join(OUT_DIR, f"equity_{tag}.csv"), index=False)

    print(tag, "finished",
          "Final =", round(summary["final_value"], 2),
          "Sharpe =", round(sharpe, 4) if pd.notna(sharpe) else sharpe)

    return daily_ret.rename(tag), equity.rename(tag), summary


def save_plots(summary_df, all_equity):
    eq = pd.concat(all_equity, axis=1).ffill().dropna()

    plt.figure(figsize=(12, 6))
    for col in eq.columns:
        if col == "buyhold_equal":
            plt.plot(eq.index, eq[col], linewidth=2.5, linestyle="--", label=col)
        else:
            plt.plot(eq.index, eq[col], linewidth=0.9, alpha=0.7, label=col)

    plt.title("Equity Curves: Buy-and-Hold vs Trading Strategies")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value, normalized")
    plt.legend(fontsize=7, ncol=3)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "equity_compare.png"), dpi=150)
    plt.close()

    top5 = (
        summary_df
        .dropna(subset=["sharpe"])
        .sort_values("sharpe", ascending=False)
        .head(5)
    )

    plt.figure(figsize=(8, 4))
    plt.barh(top5["strategy"], top5["sharpe"])
    plt.xlabel("Sharpe Ratio")
    plt.title("Top 5 Strategies by Sharpe Ratio")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "top5_sharpe.png"), dpi=150)
    plt.close()

    ma_rows = summary_df[
        summary_df["signal_col"].astype(str).str.startswith("signal_ma_")
    ].copy()

    if len(ma_rows) > 0:
        ma_rows[["short", "long"]] = (
            ma_rows["signal_col"]
            .str.extract(r"signal_ma_(\d+)_(\d+)")
            .astype(int)
        )

        pivot = ma_rows.pivot_table(
            index="short",
            columns="long",
            values="sharpe"
        )

        plt.figure(figsize=(5, 4))
        plt.imshow(pivot, aspect="auto")
        plt.colorbar(label="Sharpe Ratio")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.iloc[i, j]
                if pd.notna(value):
                    plt.text(j, i, round(value, 3), ha="center", va="center")

        plt.xlabel("Long MA Window")
        plt.ylabel("Short MA Window")
        plt.title("MA Strategy Sharpe Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "ma_heatmap.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    csvs = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csvs:
        raise FileNotFoundError("No CSV files found in data_with_signals.")

    sample_path = os.path.join(DATA_DIR, csvs[0])
    cols = pd.read_csv(sample_path).columns.tolist()

    momentum_cols = sorted([c for c in cols if c.startswith("signal_momentum_")])
    ma_cols = sorted([c for c in cols if c.startswith("signal_ma_")])
    combined_cols = sorted([c for c in cols if c.startswith("signal_combined_")])
    signal_cols = momentum_cols + ma_cols + combined_cols

    # Add buy-and-hold signal column if missing
    bh_col = "_bh_signal_"
    for f in csvs:
        path = os.path.join(DATA_DIR, f)
        df = pd.read_csv(path)
        if bh_col not in df.columns:
            df[bh_col] = 1
            df.to_csv(path, index=False)

    all_returns = []
    all_equity = []
    summaries = []

    print("\nRunning Buy and Hold benchmark...")
    bh_ret, bh_equity, bh_summary = run_backtest(
        bh_col,
        "buyhold_equal",
        is_benchmark=True
    )
    all_returns.append(bh_ret)
    all_equity.append(bh_equity)
    summaries.append(bh_summary)

    print("\nRunning signal strategies...")
    for signal_col in signal_cols:
        ret, equity, summary = run_backtest(signal_col, signal_col)
        all_returns.append(ret)
        all_equity.append(equity)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(OUT_DIR, "summary_all.csv"), index=False)

    returns_df = pd.concat(all_returns, axis=1).sort_index()
    returns_df.index.name = "date"
    returns_df.to_csv(os.path.join(OUT_DIR, "daily_returns_all.csv"))

    save_plots(summary_df, all_equity)

    print("\nSaved:")
    print("- summary_all.csv")
    print("- daily_returns_all.csv")
    print("- equity_compare.png")
    print("- top5_sharpe.png")
    print("- ma_heatmap.png")
    print("\nBacktest-only step finished.")
