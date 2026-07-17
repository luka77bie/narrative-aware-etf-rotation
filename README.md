# Narrative-Aware ETF Rotation

[English](README.md) | [简体中文](README.zh-CN.md)

A reproducible quantitative research platform for evaluating momentum,
risk controls, market-attention signals, and policy narratives across a
diversified Chinese ETF universe.

The project separates three types of evidence:

1. **Formal quantitative signals** derived from point-in-time ETF market data.
2. **Exploratory narrative proxies** derived from observable market activity.
3. **Validation-only policy and news pipelines** with explicit provenance and
   time-alignment controls.

---

## Research Question

> Can narrative information and risk controls improve the risk-adjusted
> performance of a traditional ETF momentum rotation strategy without
> introducing look-ahead bias, unstable parameter selection, or excessive
> turnover?

---

## Research Decision

The selected primary model is:

> **MOM60 monthly Top-3 ETF rotation**

The model combining MOM60 with a 50% market-attention proxy improved
full-sample performance, but did not outperform MOM60 in aggregate
walk-forward out-of-sample evaluation.

Current status:

| Component | Status |
|---|---|
| MOM60 | Selected primary model |
| MOM60 + 50% Market Attention Proxy | Exploratory candidate |
| 10–30% Proxy variants | Rejected |
| Risk Engine | Evaluated |
| Walk-Forward Validation | Complete |
| Reporting Pipeline | Complete |
| Policy Narrative V1 | Validation only |
| Narrative Signal V2 | Validation only |
| Real historical news | Not yet integrated |

Policy-derived signals are excluded from formal historical performance claims
until a complete point-in-time archive is available.

---

## Main Results

### Full-Sample Evaluation

Evaluation period:

```text
2019-10-08 to 2026-07-13
```

Assumptions:

- Top 3 ETFs
- Equal-weight portfolio
- Monthly rebalance
- Next-trading-day execution
- Transaction cost: 10 basis points multiplied by turnover
- Minimum eligible universe: 10 ETFs

| Model | CAGR | Volatility | Sharpe | Sortino | Max Drawdown | Calmar | Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| MOM60 | 14.93% | 27.50% | 0.644 | 0.861 | -45.70% | 0.327 | 45.73% |
| MOM60 + 10% Proxy | 12.58% | 27.55% | 0.568 | 0.756 | -47.53% | 0.265 | 46.14% |
| MOM60 + 20% Proxy | 12.42% | 27.47% | 0.564 | 0.743 | -45.85% | 0.271 | 45.73% |
| MOM60 + 30% Proxy | 13.52% | 27.41% | 0.600 | 0.791 | -48.12% | 0.281 | 46.55% |
| MOM60 + 50% Proxy | 15.83% | 27.01% | 0.680 | 0.913 | -42.26% | 0.375 | 49.80% |

The 50% proxy model improved the full-sample result, but the improvement was
not stable across subperiods.

### Subperiod Robustness

| Model | Period | CAGR | Sharpe | Max Drawdown | Calmar |
|---|---|---:|---:|---:|---:|
| MOM60 | Pre-2022 | 15.52% | 0.653 | -23.69% | 0.655 |
| MOM60 + 50% Proxy | Pre-2022 | 12.46% | 0.577 | -24.47% | 0.509 |
| MOM60 | 2022–2023 | -15.11% | -0.727 | -30.09% | -0.502 |
| MOM60 + 50% Proxy | 2022–2023 | -8.20% | -0.327 | -28.41% | -0.289 |
| MOM60 | 2024+ | 45.50% | 1.344 | -21.49% | 2.117 |
| MOM60 + 50% Proxy | 2024+ | 43.03% | 1.286 | -22.45% | 1.917 |

The proxy showed useful defensive behaviour during 2022–2023, but
underperformed MOM60 before 2022 and after 2024.

### Walk-Forward Out-of-Sample Evaluation

Configuration:

```text
Training context: 36 months
Test window:     12 months
Step:            12 months
Folds:           4
```

The proxy weight was fixed before walk-forward evaluation and was not
re-optimised inside individual folds.

| Model | Folds | OOS CAGR | OOS Sharpe | OOS Sortino | OOS Max Drawdown | OOS Calmar |
|---|---:|---:|---:|---:|---:|---:|
| MOM60 | 4 | 22.21% | 0.859 | 1.196 | -24.34% | 0.913 |
| MOM60 + 50% Proxy | 4 | 21.46% | 0.838 | 1.162 | -25.26% | 0.850 |

The aggregate OOS result supports MOM60 as the selected primary model.

---

## System Architecture

```text
ETF Universe
    ↓
Historical Market Data
    ↓
Data Validation and Preprocessing
    ↓
Momentum Signal
    ↓
Risk Metrics and Risk Controls
    ↓
Portfolio Construction
    ↓
Monthly Next-Day Backtest
    ↓
Benchmark and Ablation Analysis
    ↓
Walk-Forward OOS Validation
    ↓
Automated Charts
    ↓
Markdown and HTML Research Report
```

The policy and news branch is maintained separately:

```text
Official Source
    ↓
Source Registry
    ↓
Approval and Provenance Validation
    ↓
Publication and Retrieval Timestamps
    ↓
Point-in-Time Availability
    ↓
Daily Theme Aggregation
    ↓
Theme-to-ETF Mapping
    ↓
Validation-Only Policy Narrative Signal
```

---

## Momentum Baseline

The selected momentum model uses 60-trading-day price momentum:

```text
MOM60 = adjusted_close(t) / adjusted_close(t - 60) - 1
```

At each rebalance date:

1. Eligible ETFs are ranked cross-sectionally.
2. The strongest three ETFs are selected.
3. Portfolio weights are assigned equally.
4. Orders are executed on the next available trading day.
5. Transaction costs are applied according to portfolio turnover.

---

## Market-Attention Proxy

The market-attention proxy is derived from historical ETF market data rather
than historical news text.

Features include:

- short-versus-long turnover growth
- short-versus-long volume growth
- turnover attention momentum
- volatility expansion
- cross-sectional normalisation

Composite score:

```text
Composite Score
    = (1 - proxy_weight) × Z(MOM60)
    + proxy_weight × Z(Market Attention Proxy)
```

The 50% proxy model remains useful for defensive-regime analysis, but is not
the selected primary strategy.

---

## Risk Engine

The risk subsystem includes:

- rolling volatility
- downside volatility
- cross-sectional risk penalties
- absolute momentum cash allocation
- market-regime filters
- volatility targeting
- residual cash allocation
- transaction-cost-aware evaluation

Defensive overlays are retained as research components. They are not
automatically promoted when lower drawdown is accompanied by a material
reduction in return.

---

## Point-in-Time Controls

### Trading signals

```text
Signal date = t
Execution date > t
```

Same-day signal execution is prohibited.

### Policy and news records

```text
available_at = max(published_at, retrieved_at)
```

A record cannot influence a signal before both publication and retrieval have
occurred.

If a document is retrieved after market close, its effective signal date is
deferred.

The audit checks:

- duplicate logical keys
- publication timing
- retrieval timing
- availability timing
- signal cut-off time
- next-day execution
- overlapping OOS observations

---

## Repository Structure

```text
.
├── .github/
│   └── workflows/              GitHub Actions CI
├── config/                     ETF universe and strategy configuration
├── data/
│   ├── raw/                    Local historical data and cache
│   ├── processed/              Generated processed datasets
│   ├── sample/                 Committed offline fixtures
│   └── templates/              Policy and news metadata templates
├── docs/                       Research decisions and source reviews
├── outputs/
│   └── reporting/
│       ├── charts/
│       ├── research_report.md
│       └── research_report.html
├── scripts/                    Command-line research runners
├── src/
│   ├── backtest/
│   ├── data/
│   ├── evaluation/
│   ├── narrative/
│   ├── portfolio/
│   ├── reporting/
│   ├── risk/
│   └── signals/
├── tests/                      Unit and integration tests
├── main.py                     Single-command pipeline
├── requirements.txt
└── README.md
```

---

## Installation

### Clone the repository

```bash
git clone git@github.com:luka77bie/narrative-aware-etf-rotation.git
cd narrative-aware-etf-rotation
```

### Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
```

Python 3.9 or later is required.

---

## Run Tests

Run the complete test suite:

```bash
python3 -m pytest -v
```

Compile all Python sources:

```bash
python3 -m compileall -q main.py src scripts tests
```

---

## Single-Command Pipeline

List all registered stages:

```bash
python3 main.py --list-steps
```

Run the complete pipeline:

```bash
python3 main.py
```

Skip tests:

```bash
python3 main.py --skip-tests
```

Resume from a named stage:

```bash
python3 main.py \
  --skip-tests \
  --from-step "Generate research charts"
```

The complete pipeline requires local historical ETF data under `data/raw/`.

---

## Key Commands

### Momentum signal

```bash
python3 scripts/run_momentum_signal.py
python3 scripts/run_momentum_backtest.py
```

### Market-attention proxy

```bash
python3 scripts/run_narrative_proxy_signal.py
python3 scripts/run_proxy_composite_ablation.py
python3 scripts/run_proxy_robustness.py
```

### Walk-forward validation

```bash
python3 scripts/run_walk_forward_validation.py
```

### Policy validation

```bash
python3 scripts/run_policy_signal_v1.py --validation-only
python3 scripts/run_narrative_signal_v2.py --validation-only
python3 scripts/audit_narrative_time_alignment.py
```

### Reporting

```bash
python3 scripts/generate_research_charts.py
python3 scripts/generate_research_report.py
python3 scripts/generate_research_html.py
```

Open the HTML report on macOS:

```bash
open outputs/reporting/research_report.html
```

---

## Generated Outputs

Quantitative outputs:

```text
outputs/momentum_signal_history.csv
outputs/narrative_proxy_signal_history.csv
outputs/proxy_composite_ablation_metrics.csv
outputs/proxy_robustness_metrics.csv
outputs/walk_forward_fold_metrics.csv
outputs/walk_forward_aggregate_metrics.csv
outputs/walk_forward_oos_returns.csv
```

Reporting outputs:

```text
outputs/reporting/model_summary.csv
outputs/reporting/robustness_summary.csv
outputs/reporting/walk_forward_summary.csv
outputs/reporting/research_report.md
outputs/reporting/research_report.html
outputs/reporting/charts/nav_comparison.png
outputs/reporting/charts/drawdown_comparison.png
outputs/reporting/charts/walk_forward_sharpe.png
outputs/reporting/charts/subperiod_cagr.png
```

---

## Data Behaviour

The market-data loader follows:

```text
Local raw cache
    ↓
Committed sample fallback
    ↓
Explicit failure
```

The system does not silently fabricate historical market prices.

Committed sample files support:

- unit tests
- CI
- offline software validation
- API-failure recovery tests

They are not substitutes for the complete research dataset.

---

## Reproducibility Controls

The repository implements:

- stable ETF identifiers
- committed sample fixtures
- cache-first market-data loading
- explicit data validation
- duplicate-date detection
- stale-price checks
- transaction-cost modelling
- next-day execution
- minimum universe coverage
- explicit backtest start dates
- non-overlapping OOS windows
- fixed parameters during OOS evaluation
- no post-OOS proxy-weight optimisation
- source provenance records
- point-in-time availability
- market-close alignment
- automated reporting outputs

---

## Continuous Integration

GitHub Actions runs:

- Python 3.9 tests
- Python 3.12 tests
- reporting smoke tests
- Python source compilation
- pipeline registry checks
- committed fixture checks

The CI workflow does not download live market data or run the complete
historical research pipeline.

---

## Limitations

- Historical performance does not guarantee future performance.
- The ETF universe has launch-date and survivorship constraints.
- Some ETFs have shorter histories than broad-market ETFs.
- The market-attention proxy is not direct news sentiment.
- Current policy records are not a complete point-in-time archive.
- Taxes, bid-ask spreads and market impact are simplified.
- Equal-weight Top-3 portfolios have concentration risk.
- Walk-forward analysis contains a limited number of folds.
- Parameter choices remain dependent on the available historical sample.
- Real historical news is not yet integrated into formal backtests.

---

## Roadmap

Planned extensions include:

- approved historical news ingestion
- longer point-in-time policy archives
- expanding-window walk-forward analysis
- benchmark-relative attribution
- liquidity and concentration constraints
- bootstrap confidence intervals
- factor exposure analysis
- automated release artifacts
- interactive research dashboard
- scheduled data refresh and report generation

---

## Current Allocation Report

The project can generate the latest MOM60 Top-3 model allocation
from the most recent valid ETF cross-section.

Run:

```bash
python3 scripts/run_momentum_signal.py
python3 scripts/generate_current_allocation.py
```

Open the standalone allocation report on macOS:

```bash
open outputs/reporting/current_allocation.html
```

The allocation report contains:

- latest valid market-data date
- current Top-3 ETF codes and names
- MOM60 values and momentum scores
- equal target weights
- previous target allocation
- added, removed and retained ETFs
- estimated one-way turnover
- stale-data warning
- next-trading-day execution policy

Generated files:

```text
outputs/reporting/current_allocation.csv
outputs/reporting/current_allocation.md
outputs/reporting/current_allocation.html
```

The output is a model-generated research signal, not an automatic
trade instruction or investment recommendation.

---

## Disclaimer

This repository is for research and educational purposes only.

It does not constitute:

- investment advice
- a recommendation
- an offer to buy or sell securities
- a claim of future performance

All results should be independently validated before real-world use.
