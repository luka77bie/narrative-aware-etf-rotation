# Risk Engine V1 Decision

## Selected Research Baseline

- Signal: MOM60
- Portfolio: Top 3 equal weight
- Rebalance: Monthly
- Execution: Next trading-day close
- Transaction cost: 10 bps multiplied by turnover
- Minimum eligible ETF coverage: 10
- Risky exposure: 100%

## Selected Defensive Variant

- Base strategy: MOM60 Top 3
- Volatility target: 10% annualised
- Volatility lookback: 60 trading days
- Exposure uses lagged realised volatility
- Residual allocation: 159001 money-market ETF

## Results

| Model | CAGR | Sharpe | Sortino | Maximum Drawdown | Calmar | Average Risky Exposure |
|---|---:|---:|---:|---:|---:|---:|
| MOM60 Baseline | 14.933% | 0.644 | 0.861 | -45.698% | 0.327 | 100.000% |
| 10% Volatility Target | 7.078% | 0.651 | 0.886 | -25.767% | 0.275 | 47.472% |
| 15% Volatility Target | 9.745% | 0.637 | 0.862 | -36.054% | 0.270 | 67.921% |
| 20% Volatility Target | 11.846% | 0.642 | 0.871 | -42.397% | 0.279 | 82.139% |

## Decision

The MOM60 strategy remains the primary baseline because it has the highest CAGR and Calmar ratio.

The 10% volatility-targeting variant is retained as a defensive profile because it materially reduces maximum drawdown and slightly improves Sharpe and Sortino ratios. It does not replace the baseline because the reduction in CAGR is substantial.

The 15% and 20% volatility targets are not retained as preferred variants because neither improves Sharpe or Calmar relative to the baseline.

Cross-sectional risk penalties, slot-level cash filtering, and the CSI 300 regime filter are retained as negative ablation results.
