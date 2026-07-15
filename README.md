# Narrative-Aware ETF Rotation

A reproducible research pipeline for evaluating momentum, risk
controls and market-attention signals across a diversified Chinese
ETF universe.

The project separates three types of evidence:

1. **Formal quantitative signals** derived from point-in-time ETF
   market data.
2. **Exploratory narrative proxies** derived from turnover, volume
   and volatility.
3. **Validation-only policy/news pipelines** with strict provenance
   and availability controls.

## Research Decision

The selected primary model is:

> **MOM60 monthly Top-3 ETF rotation**

The 50% market-attention proxy improved full-sample results, but did
not outperform MOM60 in aggregate walk-forward out-of-sample
evaluation. It therefore remains exploratory.

Policy-derived narrative signals remain validation-only and are not
used in formal historical performance claims.

## Main Results

### Full-Sample Evaluation

Evaluation period: 2019-10-08 to 2026-07-13.

| Model | CAGR | Sharpe | Sortino | Max Drawdown | Calmar |
|---|---:|---:|---:|---:|---:|
| MOM60 | 14.93% | 0.644 | 0.861 | -45.70% | 0.327 |
| MOM60 + 50% Proxy | 15.83% | 0.680 | 0.913 | -42.26% | 0.375 |

### Walk-Forward Out-of-Sample Evaluation

| Model | Folds | CAGR | Sharpe | Sortino | Max Drawdown | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| MOM60 | 4 | 22.21% | 0.859 | 1.196 | -24.34% | 0.913 |
| MOM60 + 50% Proxy | 4 | 21.46% | 0.838 | 1.162 | -25.26% | 0.850 |

The OOS result supports MOM60 as the primary research baseline.

## Strategy Configuration

- Universe: 21 diversified Chinese ETFs
- Primary signal: 60-day momentum
- Portfolio: Top 3 equal weight
- Rebalance: Monthly
- Execution: Next trading day
- Transaction cost: 10 bps multiplied by turnover
- Minimum eligible assets: 10
- Cash instrument: Money-market ETF
- Risk controls:
  - absolute momentum
  - market-regime filtering
  - volatility targeting
  - risk-adjusted scoring

## Market-Attention Proxy

The proxy is derived only from historical ETF market data:

- short/long turnover growth
- short/long volume growth
- turnover attention momentum
- volatility expansion

It is not described as historical news or sentiment data.

## Policy and News Data Controls

The policy/news subsystem includes:

- source registry
- source-approval evidence
- URL and article provenance
- stable document identifiers
- duplicate detection
- publication and retrieval timestamps
- point-in-time availability
- market-close alignment
- theme-to-ETF mapping

Manually collected policy records become available only after:

```text
available_at = max(published_at, retrieved_at)
config/                 ETF universe, themes and strategy profiles
data/sample/            Small committed offline fixtures
data/raw/               Local market and source data; not committed
docs/                   Research decisions and source reviews
outputs/                Generated backtests, reports and charts
scripts/                Command-line research runners
src/backtest/           Backtest engine
src/data/               Market-data loading and validation
src/evaluation/         Benchmark and walk-forward evaluation
src/narrative/          Proxy, policy and provenance components
src/portfolio/          Portfolio construction
src/reporting/          Reporting and chart generation
src/risk/               Risk metrics and overlays
src/signals/            Momentum signal implementation
tests/                  Unit and integration tests
main.py                 Single-command pipeline
git clone git@github.com:luka77bie/narrative-aware-etf-rotation.git
cd narrative-aware-etf-rotation

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pytest -v
cat > README.md << 'EOF'
# Narrative-Aware ETF Rotation

A reproducible quantitative research platform for evaluating momentum,
risk controls, market-attention signals, and policy narratives across a
diversified Chinese ETF universe.

The project is designed around a strict separation between:

1. **Formal quantitative signals** derived from historical ETF market data.
2. **Exploratory narrative proxies** derived from observable market attention.
3. **Validation-only policy and news pipelines** with explicit provenance and
   point-in-time controls.

---

## Research Question

> Can narrative information and risk controls improve the risk-adjusted
> performance of a traditional ETF momentum rotation strategy without
> introducing look-ahead bias, unstable parameter selection, or excessive
> turnover?

---

## Current Research Decision

The selected primary model is:

> **MOM60 monthly Top-3 ETF rotation**

The model combining MOM60 with a 50% market-attention proxy improved
full-sample metrics, but did not outperform MOM60 in aggregate walk-forward
out-of-sample evaluation.

Accordingly:

| Component | Research status |
|---|---|
| MOM60 | Selected primary model |
| MOM60 + 50% Market Attention Proxy | Exploratory candidate |
| 10–30% Proxy variants | Rejected |
| Policy Narrative V1 | Pipeline validation only |
| Narrative Signal V2 | Pipeline validation only |
| Historical news ingestion | Infrastructure complete; real archive not yet integrated |

No additional proxy-weight optimisation is performed after observing the
out-of-sample results.

---

## Main Results

### Full-Sample Evaluation

Evaluation period:

```text
2019-10-08 to 2026-07-13
```

Portfolio assumptions:

- Top 3 ETFs
- Equal weight
- Monthly rebalance
- Next-trading-day execution
- Transaction cost: 10 basis points multiplied by turnover
- Minimum eligible universe: 10 ETFs

| Model | CAGR | Annual volatility | Sharpe | Sortino | Maximum drawdown | Calmar | Average turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| MOM60 | 14.93% | 27.50% | 0.644 | 0.861 | -45.70% | 0.327 | 45.73% |
| MOM60 + 10% Proxy | 12.58% | 27.55% | 0.568 | 0.756 | -47.53% | 0.265 | 46.14% |
| MOM60 + 20% Proxy | 12.42% | 27.47% | 0.564 | 0.743 | -45.85% | 0.271 | 45.73% |
| MOM60 + 30% Proxy | 13.52% | 27.41% | 0.600 | 0.791 | -48.12% | 0.281 | 46.55% |
| MOM60 + 50% Proxy | 15.83% | 27.01% | 0.680 | 0.913 | -42.26% | 0.375 | 49.80% |

The 50% proxy variant improved the full-sample result, but this improvement
was not stable across subperiods.

### Subperiod Robustness

| Model | Period | CAGR | Sharpe | Maximum drawdown | Calmar |
|---|---|---:|---:|---:|---:|
| MOM60 | Pre-2022 | 15.52% | 0.653 | -23.69% | 0.655 |
| MOM60 + 50% Proxy | Pre-2022 | 12.46% | 0.577 | -24.47% | 0.509 |
| MOM60 | 2022–2023 | -15.11% | -0.727 | -30.09% | -0.502 |
| MOM60 + 50% Proxy | 2022–2023 | -8.20% | -0.327 | -28.41% | -0.289 |
| MOM60 | 2024+ | 45.50% | 1.344 | -21.49% | 2.117 |
| MOM60 + 50% Proxy | 2024+ | 43.03% | 1.286 | -22.45% | 1.917 |

The proxy provided useful defensive behaviour during the weak 2022–2023
period, but underperformed MOM60 before 2022 and after 2024.

### Walk-Forward Out-of-Sample Evaluation

Configuration:

```text
Training context: 36 months
Test window:     12 months
Step:            12 months
Folds:           4
```

The proxy weight was fixed before walk-forward evaluation. It was not
re-optimised inside individual folds.

| Model | Folds | OOS CAGR | OOS Sharpe | OOS Sortino | OOS maximum drawdown | OOS Calmar |
|---|---:|---:|---:|---:|---:|---:|
| MOM60 | 4 | 22.21% | 0.859 | 1.196 | -24.34% | 0.913 |
| MOM60 + 50% Proxy | 4 | 21.46% | 0.838 | 1.162 | -25.26% | 0.850 |

The aggregate walk-forward result supports MOM60 as the selected primary
model.

---

## Strategy Architecture

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
Automated Charts and Research Report
```

The policy/news branch is maintained separately:

```text
Official source
    ↓
Source registry and approval evidence
    ↓
Provenance validation
    ↓
Publication and retrieval timestamps
    ↓
Point-in-time availability
    ↓
Daily theme aggregation
    ↓
Theme-to-ETF mapping
    ↓
Validation-only Policy Narrative Signal
```

---

## ETF Universe

The official universe contains 21 unique instruments across:

- Broad market
  - CSI 300
  - CSI 500
  - CSI 1000
  - ChiNext
  - STAR 50
- Technology
  - Artificial intelligence
  - Communications
  - Semiconductors
  - Robotics
  - Computers
- Commodity and cyclical exposures
  - Gold
  - Nonferrous metals
  - Rare earths
  - Coal
- Defensive and value exposures
  - Dividend
  - Low volatility
  - Banks
  - Power utilities
- Healthcare and consumption
  - Healthcare
  - Consumer electronics
- Cash
  - Money-market ETF

The universe builder is cache-first and fault-tolerant. Temporary API
failures do not silently alter stable universe membership.

---

## Momentum Baseline

The primary momentum model uses 60-trading-day price momentum:

```text
MOM60 = adjusted_close(t) / adjusted_close(t - 60) - 1
```

At each rebalance date:

1. Eligible ETFs are ranked cross-sectionally.
2. The strongest three ETFs are selected.
3. Portfolio weights are assigned equally.
4. Orders are executed on the next available trading day.
5. Transaction costs are proportional to portfolio turnover.

The research also evaluates MOM20 and combined momentum specifications, but
MOM60 remains the selected baseline.

---

## Market-Attention Proxy

The market-attention proxy is derived entirely from point-in-time ETF market
data. It is not presented as historical news sentiment.

Its features include:

- short-versus-long turnover growth
- short-versus-long volume growth
- turnover attention momentum
- volatility expansion
- cross-sectional normalisation
- theme-aware ETF ranking

The combined score is:

```text
Composite Score
    = (1 - proxy_weight) × Z(MOM60)
    + proxy_weight × Z(Market Attention Proxy)
```

The 50% proxy model is retained for explanatory and defensive-regime
analysis, but not selected as the primary strategy.

---

## Risk Engine

The risk subsystem includes:

- rolling volatility
- downside volatility
- cross-sectional risk penalties
- absolute momentum cash allocation
- market-regime filters
- volatility targeting
- cash residual allocation
- transaction-cost-aware evaluation

Several defensive overlays reduced exposure or drawdown, but also reduced
returns materially. They are therefore retained as research components
rather than automatically promoted into the selected model.

---

## Point-in-Time Controls

The project enforces explicit time-alignment rules.

### Trading signals

```text
Signal date = t
Execution date > t
```

Same-day execution is prohibited.

### Policy and news records

```text
available_at = max(published_at, retrieved_at)
```

A record cannot influence a signal before both publication and retrieval
have occurred.

If a document is retrieved after the configured market close, its effective
signal date is deferred.

The audit checks:

- duplicate logical keys
- publication timing
- retrieval timing
- availability timing
- signal cut-off time
- next-day execution
- overlapping OOS observations

---

## Policy and News Data Status

The repository contains infrastructure for:

- official-source registration
- source review evidence
- research-use approval controls
- article and document identifiers
- canonical URL handling
- duplicate detection
- source-category weighting
- theme classification
- daily policy/news features
- point-in-time filtering
- theme-to-ETF mapping

Current manually collected policy records are used only to validate the
software pipeline.

They are not treated as a complete historical archive and are excluded from
formal historical performance claims.

No synthetic news record is permitted in a formal research dataset.

---

## Repository Structure

```text
.
├── .github/
│   └── workflows/                 GitHub Actions workflows
├── config/
│   ├── etf_universe.csv           Official ETF universe
│   ├── narrative_themes.csv       Narrative taxonomy
│   ├── narrative_profiles.csv     Selected/rejected model profiles
│   ├── news_sources.csv           Source registry
│   └── news_column_maps/          External-source field mappings
├── data/
│   ├── raw/                       Local downloaded data and caches
│   ├── processed/                 Generated processed datasets
│   ├── sample/                    Committed offline test fixtures
│   └── templates/                 News and policy metadata templates
├── docs/
│   ├── narrative_signal_v1_decision.md
│   ├── risk_engine_decision.md
│   ├── walk_forward_validation_v1_decision.md
│   └── source_reviews/
├── outputs/
│   └── reporting/
│       ├── charts/
│       ├── research_report.md
│       └── research_report.html
├── scripts/
│   ├── build_official_etf_universe.py
│   ├── run_momentum_signal.py
│   ├── run_momentum_backtest.py
│   ├── run_narrative_proxy_signal.py
│   ├── run_proxy_composite_ablation.py
│   ├── run_proxy_robustness.py
│   ├── run_walk_forward_validation.py
│   ├── generate_research_charts.py
│   ├── generate_research_report.py
│   └── generate_research_html.py
├── src/
│   ├── backtest/                  Backtest engine
│   ├── data/                      Market-data loading and validation
│   ├── evaluation/                Benchmark and walk-forward evaluation
│   ├── narrative/                 Proxy, policy and provenance modules
│   ├── portfolio/                 Portfolio construction
│   ├── reporting/                 Reporting and chart generation
│   ├── risk/                      Risk metrics and overlays
│   └── signals/                   Momentum signals
├── tests/                         Unit and integration tests
├── main.py                        Single-command pipeline
├── requirements.txt
└── README.md
```

Generated raw and processed research datasets are generally excluded from
version control. Small sample fixtures are committed for offline testing.

---

## Installation

### 1. Clone the repository

```bash
git clone git@github.com:luka77bie/narrative-aware-etf-rotation.git
cd narrative-aware-etf-rotation
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade packaging tools

```bash
python3 -m pip install --upgrade pip setuptools wheel
```

### 4. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

The project requires Python 3.9 or later.

For a clean environment, the dependency set includes packages used for:

- tabular processing
- numerical calculations
- HTTP access
- market-data retrieval
- testing
- chart generation
- Markdown/HTML reporting

---

## Running Tests

Run the complete test suite:

```bash
python3 -m pytest -v
```

Run quietly:

```bash
python3 -m pytest -q
```

Compile all Python sources:

```bash
python3 -m compileall -q main.py src scripts tests
```

---

## Single-Command Pipeline

List available stages:

```bash
python3 main.py --list-steps
```

The registered pipeline stages are:

1. Run test suite
2. Build momentum signal
3. Build market-attention proxy
4. Run proxy ablation
5. Run proxy robustness analysis
6. Run walk-forward validation
7. Generate research charts
8. Generate research report
9. Generate HTML report

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

Generate only the report and subsequent outputs:

```bash
python3 main.py \
  --skip-tests \
  --from-step "Generate research report"
```

The full pipeline requires local historical ETF data under `data/raw/`.

---

## Key Individual Commands

### Momentum

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

Open the generated HTML report on macOS:

```bash
open outputs/reporting/research_report.html
```

---

## Generated Outputs

Important quantitative outputs include:

```text
outputs/momentum_signal_history.csv
outputs/narrative_proxy_signal_history.csv
outputs/proxy_composite_ablation_metrics.csv
outputs/proxy_robustness_metrics.csv
outputs/walk_forward_fold_metrics.csv
outputs/walk_forward_aggregate_metrics.csv
outputs/walk_forward_oos_returns.csv
```

Reporting outputs include:

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

## Offline and Fault-Tolerant Data Behaviour

The market-data loader follows:

```text
local raw cache
    ↓
committed sample fallback
    ↓
explicit failure
```

It does not silently generate market prices.

The sample files support:

- unit tests
- offline software validation
- CI
- API-failure recovery tests

They are not substitutes for the complete historical research dataset.

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
- explicit backtest start date
- non-overlapping OOS test windows
- fixed parameters during OOS evaluation
- no post-OOS weight optimisation
- source provenance records
- point-in-time document availability
- market-close alignment
- deterministic reporting outputs

---

## Git Workflow

Development is organised through feature branches and pull requests.

Typical workflow:

```bash
git checkout main
git pull origin main

git checkout -b feature/example-v1
git push -u origin feature/example-v1

# make changes

git add .
git commit -m "feat: describe the change"
git push
```

After review and merge:

```bash
git checkout main
git pull origin main
git branch -d feature/example-v1
git fetch --prune
```

---

## Limitations

- Historical performance does not guarantee future performance.
- The ETF universe contains launch-date and survivorship constraints.
- Some ETFs have substantially shorter histories than broad-market ETFs.
- The market-attention proxy is not a direct measure of news sentiment.
- Current policy metadata is not a complete point-in-time archive.
- Taxes, bid-ask spread and market impact are simplified.
- Equal-weight Top-3 portfolios may have material concentration risk.
- Walk-forward results are based on a limited number of folds.
- Parameter choices remain dependent on the available historical sample.
- Real historical news has not yet been integrated into formal backtests.

---

## Roadmap

Planned extensions include:

- licensed or approved historical news ingestion
- longer point-in-time policy archives
- expanding-window walk-forward analysis
- benchmark-relative attribution
- concentration and liquidity constraints
- bootstrap confidence intervals
- factor exposure analysis
- automated release artifacts
- interactive research dashboard
- scheduled data refresh and report generation

---

## Disclaimer

This repository is for research and educational purposes only.

It does not constitute:

- investment advice
- a recommendation
- an offer to buy or sell securities
- a claim of future performance

All results should be independently validated before any real-world use.
