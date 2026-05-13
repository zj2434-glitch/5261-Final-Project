# 5261 Final Project

This project studies whether simple trend-following strategies can generate useful risk-adjusted returns in U.S. tech stocks.

We use daily price data from **2016-01-01 to 2025-09-30** and compare:

- momentum strategies
- moving-average strategies
- combined strategies
- an equal-weight buy-and-hold benchmark

The project includes backtesting, statistical inference, and robustness analysis.

## Data

- Source: Yahoo Finance (`yahooquery`)
- Stocks: 15 U.S. tech companies across large-, mid-, and small-cap groups
- Sample period: **2016-01-01 to 2025-09-30**

## Strategy Setup

We test:

- Momentum: 30 / 60 / 90-day signals
- Moving Average: (10,30), (20,100), (50,200)
- Combined: momentum + moving average

All strategies are run in an **equal-weight, long-only** framework with:

- initial capital: \$100,000
- transaction cost: 0.1%

## Files

- `5261_step1_data(1).py`  
  Download and clean price data

- `5261_step2_signals(1).py`  
  Generate momentum, MA, and combined signals

- `5261_step3_backtest_only(1).py`  
  Run backtests and save main performance outputs

- `5261_step4_inference(1).py`  
  Run one-sample t-tests, paired t-tests, and 95% confidence intervals

- `5261_step5_robustness(1).py`  
  Run sub-period analysis and bootstrap Sharpe confidence intervals

## How to Run

Run the scripts in this order:

```bash
python 5261_step1_data(1).py
python 5261_step2_signals(1).py
python 5261_step3_backtest_only(1).py
python 5261_step4_inference(1).py
python 5261_step5_robustness(1).py
```

## Main Outputs

### `bt_outputs/`
- `summary_all.csv`
- `daily_returns_all.csv`
- `equity_compare.png`
- `top5_sharpe.png`
- `ma_heatmap.png`

### `inference_outputs/`
- `inference_all.csv`
- `inference_best_strategy.csv`

### `robustness_outputs/`
- `robustness_subperiod_summary.csv`
- `robustness_bootstrap_ci.csv`
- `robustness_sharpe_ci.png`

## Notes

- Benchmark: **equal-weight buy-and-hold**
- Inference: **one-sample t-test + paired t-test + 95% CI**
- Robustness: **sub-period analysis + bootstrap (B = 1000)**
