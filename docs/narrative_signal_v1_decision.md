# Narrative Signal V1 Decision

## Scope

Narrative Signal V1 evaluates whether market-observable attention
features provide incremental value relative to the validated MOM60
ETF momentum baseline.

The historical market-attention proxy uses only point-in-time market
data:

- turnover growth
- volume growth
- turnover attention momentum
- volatility expansion

Official-policy metadata is maintained separately as
pipeline-validation-only data and is excluded from formal historical
performance claims.

## Primary Baseline

- Signal: MOM60
- Portfolio: Top 3 equal weight
- Rebalance: Monthly
- Execution: Next trading day
- Transaction cost: 10 basis points multiplied by turnover
- Minimum eligible assets: 10

## Proxy Ablation Results

| Model | CAGR | Sharpe | Sortino | Maximum Drawdown | Calmar | Average Turnover |
|---|---:|---:|---:|---:|---:|---:|
| MOM60 | 14.933% | 0.644 | 0.861 | -45.698% | 0.327 | 45.732% |
| MOM60 + 10% Proxy | 12.580% | 0.568 | 0.756 | -47.531% | 0.265 | 46.138% |
| MOM60 + 20% Proxy | 12.420% | 0.564 | 0.743 | -45.852% | 0.271 | 45.732% |
| MOM60 + 30% Proxy | 13.517% | 0.600 | 0.791 | -48.123% | 0.281 | 46.545% |
| MOM60 + 50% Proxy | 15.830% | 0.680 | 0.913 | -42.263% | 0.375 | 49.797% |

## Subperiod Robustness

| Model | Period | CAGR | Sharpe | Maximum Drawdown | Calmar |
|---|---|---:|---:|---:|---:|
| MOM60 | Pre-2022 | 15.523% | 0.653 | -23.688% | 0.655 |
| MOM60 + 50% Proxy | Pre-2022 | 12.461% | 0.577 | -24.468% | 0.509 |
| MOM60 | 2022-2023 | -15.112% | -0.727 | -30.094% | -0.502 |
| MOM60 + 50% Proxy | 2022-2023 | -8.198% | -0.327 | -28.405% | -0.289 |
| MOM60 | 2024+ | 45.501% | 1.344 | -21.489% | 2.117 |
| MOM60 + 50% Proxy | 2024+ | 43.032% | 1.286 | -22.451% | 1.917 |

## Decision

MOM60 remains the primary research baseline.

The 50% market-attention proxy variant is retained as an exploratory
candidate. It improves full-sample CAGR, Sharpe, Sortino, maximum
drawdown and Calmar, and materially reduces losses during 2022-2023.

However, it underperforms MOM60 during both the pre-2022 period and
the 2024+ period. Therefore, the improvement is not sufficiently
stable across subperiods to replace MOM60 as the primary strategy.

Proxy weights of 10%, 20% and 30% are rejected because they reduce
risk-adjusted performance without delivering consistent drawdown
improvement.

No additional proxy-weight optimisation is performed in order to
limit parameter overfitting.

## Policy Narrative Status

The official-policy pipeline includes:

- provenance validation
- source approval workflow
- point-in-time availability
- market-close alignment
- daily policy aggregation
- theme-to-ETF mapping

Current policy records were manually retrieved after their original
publication dates. They are therefore used only for pipeline
validation and are excluded from historical backtests.

## Selected Outputs

- Primary model: MOM60
- Exploratory model: MOM60 + 50% Market Attention Proxy
- Policy model: Pipeline validation only
