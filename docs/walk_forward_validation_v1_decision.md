# Walk-Forward Validation V1 Decision

## Objective

Evaluate the fixed MOM60 baseline and the previously identified
50% market-attention proxy candidate using non-overlapping
out-of-sample test windows.

No proxy-weight optimisation was performed inside the folds.

## Configuration

- Training context: 36 months
- Test window: 12 months
- Step: 12 months
- Portfolio: Top 3 equal weight
- Rebalance: Monthly
- Execution: Next trading day
- Minimum eligible assets: 10
- Transaction cost: 10 basis points multiplied by turnover

## Models

1. MOM60
2. MOM60 + 50% Market Attention Proxy

The proxy weight was fixed before walk-forward evaluation.

## Aggregate Out-of-Sample Results

| Model | Folds | CAGR | Sharpe | Sortino | Maximum Drawdown | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| MOM60 | 4 | 22.209% | 0.859 | 1.196 | -24.335% | 0.913 |
| MOM60 + 50% Proxy | 4 | 21.464% | 0.838 | 1.162 | -25.263% | 0.850 |

## Fold-Level Interpretation

The proxy candidate improved selected folds, particularly during
weaker market conditions, but did not provide stable improvement
across the complete out-of-sample sequence.

Both models produced positive Sharpe ratios in two of four folds.

The proxy candidate underperformed the MOM60 baseline on aggregate
out-of-sample CAGR, Sharpe, Sortino, maximum drawdown and Calmar.

## Decision

MOM60 remains the selected primary model.

The 50% market-attention proxy is retained only as an exploratory
research variant. It is not promoted into the selected production
or primary research configuration.

No further proxy-weight optimisation will be performed in this
stage, because additional tuning after observing out-of-sample
results would increase selection and overfitting risk.

## Policy Narrative Status

Policy-derived narrative signals remain pipeline-validation-only.
They are excluded from formal historical performance claims because
the current manually collected records are not a point-in-time
historical archive.

## Final Status

- Selected model: MOM60
- Exploratory variant: MOM60 + 50% Market Attention Proxy
- Rejected as primary model: Market Attention Proxy composite
- Policy Narrative: Validation only
