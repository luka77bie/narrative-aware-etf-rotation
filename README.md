# Narrative-Aware ETF Rotation

A reproducible quantitative research platform for evaluating momentum,
risk controls, market-attention signals, and policy narratives across a
diversified Chinese ETF universe.

## Research Decision

The selected primary model is **MOM60 monthly Top-3 ETF rotation**.

The 50% market-attention proxy improved full-sample performance, but did not
outperform MOM60 in aggregate walk-forward out-of-sample evaluation. It
therefore remains an exploratory candidate.

Policy-derived narrative signals remain validation-only and are excluded
from formal historical performance claims.

## Main Results

### Full-Sample Evaluation

| Model | CAGR | Sharpe | Sortino | Maximum Drawdown | Calmar |
|---|---:|---:|---:|---:|---:|
| MOM60 | 14.93% | 0.644 | 0.861 | -45.70% | 0.327 |
| MOM60 + 50% Proxy | 15.83% | 0.680 | 0.913 | -42.26% | 0.375 |

### Walk-Forward OOS Evaluation

| Model | Folds | CAGR | Sharpe | Sortino | Maximum Drawdown | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| MOM60 | 4 | 22.21% | 0.859 | 1.196 | -24.34% | 0.913 |
| MOM60 + 50% Proxy | 4 | 21.46% | 0.838 | 1.162 | -25.26% | 0.850 |

## Pipeline

```text
ETF Universe
→ Market Data
→ Validation
→ Momentum Signal
→ Risk Controls
→ Portfolio Construction
→ Backtest
→ Walk-Forward Validation
→ Charts
→ Markdown and HTML Report
git clone git@github.com:luka77bie/narrative-aware-etf-rotation.git
cd narrative-aware-etf-rotation

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
[200~git restore README.md
