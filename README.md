# FinCloud

FinCloud is a local-first Python project for financial analytics. Phase 2 adds
a modular historical daily market-data ingestion pipeline for five NSE-listed
symbols.

## Data source and limitations

The initial provider is **Yahoo Finance through `yfinance`**, used only as a
supplementary research data source. Yahoo Finance is **not an official NSE data
source**. Its data may be delayed, incomplete, adjusted, revised, or otherwise
different from exchange, broker, or other licensed datasets. Availability and
historical coverage can vary by symbol and date.

Review Yahoo Finance's current terms and applicable exchange/data-provider
terms before use. This project does not grant rights to redistribute or
commercialize downloaded data. Use the data at your own risk; it is not
investment advice. `yfinance` is an open-source wrapper around Yahoo Finance
interfaces and is not affiliated with or endorsed by Yahoo.

The provider is isolated behind a small interface in `src/ingestion/market_data.py`
so another provider can be implemented without changing the validator.

## Setup (Windows PowerShell)

Use Python 3.11. From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The virtual environment does not inherit globally installed packages. Install
the requirements into `.venv` before running the pipeline or tests. Calling its
Python executable explicitly ensures packages are installed and commands run
in the project environment, even when the environment is not activated:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Activation is optional. To activate it in PowerShell instead, use
`.\.venv\Scripts\Activate.ps1`; if script execution is blocked, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that
PowerShell session and activate again.

## Download historical data

Run from the project directory. The end date is **exclusive**, matching
yfinance's date-range behavior:

```powershell
.\.venv\Scripts\python.exe -m src.ingestion.market_data --start 2020-01-01 --end 2025-01-01
```

Download a subset, change retry behavior, or choose another local output
location:

```powershell
.\.venv\Scripts\python.exe -m src.ingestion.market_data --start 2023-01-01 --end 2024-01-01 --symbols RELIANCE TCS
.\.venv\Scripts\python.exe -m src.ingestion.market_data --start 2023-01-01 --end 2024-01-01 --max-attempts 5 --retry-delay 2
.\.venv\Scripts\python.exe -m src.ingestion.market_data --start 2023-01-01 --end 2024-01-01 --output-dir data/raw/nse --report data/processed/validation_report.json
```

Configured symbols and Yahoo Finance tickers:

| Symbol | Ticker |
|---|---|
| RELIANCE | `RELIANCE.NS` |
| TCS | `TCS.NS` |
| INFY | `INFY.NS` |
| HDFCBANK | `HDFCBANK.NS` |
| ICICIBANK | `ICICIBANK.NS` |

The pipeline requests daily OHLCV and adjusted close where supplied. It writes
CSV files and writes Parquet files only when `pyarrow` is importable. Download
errors are retried up to the configured attempt limit, recorded in the
validation report, and do not prevent attempting other configured symbols.

## Output structure

```text
data/
  raw/
    nse/
      RELIANCE.csv
      TCS.csv
      INFY.csv
      HDFCBANK.csv
      ICICIBANK.csv
      all_stocks.csv
      # Corresponding .parquet files when pyarrow is available
  processed/
    validation_report.json
```

CSV columns are `date`, `symbol`, `open`, `high`, `low`, `close`, optional
`adj_close`, and `volume`. Dates use `YYYY-MM-DD`. Each run writes fresh
snapshots to these paths rather than appending to prior downloads. Source rows
are not filled, deduplicated, or dropped; questionable rows remain available
for review. A failed symbol is not written for that run; any prior per-symbol
file is left untouched, and the report records the failed download.

## Validation and tests

The JSON report covers missing required columns and values, duplicate dates per
symbol, non-positive or non-finite prices, invalid OHLC relationships, negative
volume, and date ordering. Download failures and empty responses are also
reported. The pipeline does not silently repair or delete questionable
financial records. A nonzero command exit status indicates download or
validation issues; inspect `data/processed/validation_report.json`.

Run the offline synthetic tests (no market-data requests):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Or, after installing the requirements:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Phase 3: Silver data cleaning and preprocessing

Phase 3 reads the existing `data/raw/nse/` files only; it does not download
market data or alter the raw layer. It standardizes columns, dates, ticker
symbols, and numeric types; sorts each stock by date; validates source rows;
and produces daily returns, log returns, and 20-observation rolling volatility.

Run from the project directory with the isolated environment:

```powershell
.\.venv\Scripts\python.exe -m src.preprocessing.pipeline
```

Optional local paths and the rolling window can be specified explicitly:

```powershell
.\.venv\Scripts\python.exe -m src.preprocessing.pipeline --raw-dir data/raw/nse --output-dir data/processed/silver --rolling-window 20
```

### Silver outputs

```text
data/processed/silver/
  RELIANCE.csv
  RELIANCE.parquet
  TCS.csv
  TCS.parquet
  INFY.csv
  INFY.parquet
  HDFCBANK.csv
  HDFCBANK.parquet
  ICICIBANK.csv
  ICICIBANK.parquet
  all_stocks.csv
  all_stocks.parquet
  silver_validation_report.json
  silver_dataset_manifest.json
```

Processed rows retain source OHLCV and `adj_close` values and add
`daily_return`, `log_return`, `rolling_volatility_20d`,
`rolling_volatility_warmup`, `source_row_number`, and `quality_flags`. Adjusted
close is used for return calculations only when the source provides its
`adj_close` (or clearly equivalent adjusted-close) column; `close` is never
substituted. No prices or returns are forward-filled. Invalid or missing source
values remain missing and are flagged. `quality_flags` is a JSON array of flag
codes (`[]` means no row-level findings).

The 20-row volatility is the sample standard deviation (`ddof=1`) of 20
consecutive valid daily returns and is not annualized. The warm-up flag is true
until that full window exists. A return is calculated only when both adjusted
prices are valid and the dates are consecutive in the observed session
calendar. Session gaps are detected by comparing a ticker with the union of
dates present in the configured source tickers. Weekends and holidays are not
assumed to be sessions; dates absent from every source cannot be inferred.

Duplicate ticker/date rows are not silently discarded: the first source row
according to its recorded row number is retained, flagged, and later duplicate
rows are listed in the report's exclusions section. Other questionable rows
remain in Silver with flags. The validation report includes source parity,
data-quality, missing-session, calculation, and output parity checks. The
manifest records source and output SHA-256 checksums, row counts, date range,
runtime versions, processing configuration, and processing time. Output
financial records contain no processing timestamp.

The source adjusted-close series is used as supplied; FinCloud does not
independently establish a complete corporate-action history or guarantee
corporate-action correctness. Yahoo Finance data is supplementary research
data, not official or exchange-certified data. The dataset is not claimed to
be free of survivorship bias.

## Phase 4: Financial risk analytics

Phase 4 reads the Silver `all_stocks.csv` and uses its Phase 3
adjusted-close-based `daily_return` field. It does not recalculate returns from
unadjusted `close`, download market data, or modify raw or Silver inputs.

Run the default equal-weight, long-only portfolio with the 0% computational
risk-free fallback:

```powershell
.\.venv\Scripts\python.exe -m src.analytics.reporting --input-dir data/processed/silver --output-dir data/processed/analytics --risk-free-rate 0.0
```

Provide custom weights, a nonzero annual risk-free rate, a benchmark return CSV,
and an initial portfolio value as needed:

```powershell
.\.venv\Scripts\python.exe -m src.analytics.reporting --input-dir data/processed/silver --output-dir data/processed/analytics --weights RELIANCE=0.30 TCS=0.20 INFY=0.20 HDFCBANK=0.15 ICICIBANK=0.15 --risk-free-rate 0.05 --benchmark-csv C:\path\to\your_benchmark_returns.csv --initial-value 100000
```

Replace the example benchmark path with your own benchmark-return CSV; no
benchmark file is bundled or downloaded.

Weights must cover the selected symbols exactly, be finite, and sum to 1 within
`1e-8`. The default is long-only; negative weights require `--allow-short`.
Portfolio returns are calculated with constant configured weights and
**rebalanced daily**: each day's portfolio return is the weighted sum of that
day's component returns. This differs from buy-and-hold, where asset weights
drift with price performance.

The annual risk-free rate is an effective annual rate converted to an effective
daily rate as `(1 + annual_rate) ** (1 / 252) - 1`. The 0% default is only a
computational fallback, not an estimate of Indian government yields. Metrics
use 252 trading days per year. Annualized volatility is sample daily standard
deviation times `sqrt(252)`. Sharpe uses annualized mean daily excess return
divided by annualized asset-return volatility. Sortino uses annualized mean
excess return divided by annualized downside deviation of daily excess returns.
CAGR uses compounded total growth over actual elapsed calendar days divided by
365.25. Undefined or insufficient metrics are reported as unavailable/NaN,
not replaced with zero.

Historical VaR and CVaR use empirical loss observations (`loss = -return`),
with linearly interpolated quantile VaR and CVaR as the mean loss at or beyond
that threshold. Positive values represent losses. Drawdown compounds returns
from the configured initial portfolio value (default 1.0), tracks the running
wealth peak, and reports percentage decline from that peak.

Correlation uses pairwise-complete, date-aligned daily returns and reports
pairwise observation counts; missing returns are never filled with zero.
Benchmark beta is `Cov(asset, benchmark) / Var(benchmark)` over dates shared by
both series. To calculate beta, supply a CSV with `date` and `daily_return` (or
`return`) columns. Duplicate, missing, non-finite, and below -100% benchmark
observations are reported and excluded. Without a valid benchmark, beta is
unavailable; FinCloud does not substitute an equal-weight stock portfolio or
fabricate benchmark data. NIFTY 50 TRI integration is deferred to a future data
expansion.

### Analytics outputs

```text
data/processed/analytics/
  stock_performance_metrics.csv
  portfolio_daily_returns.csv
  portfolio_performance_metrics.json
  correlation_matrix.csv
  portfolio_drawdown.csv
  risk_metrics.csv
  analytics_validation_report.json
  analytics_manifest.json
```

The validation report records excluded source observations and reasons,
incomplete portfolio dates, correlation observation counts, and benchmark
availability. Initial missing returns and the rolling-volatility warm-up from
Silver are not imputed. The manifest records input checksums, configuration,
portfolio weights, observation counts, outputs, package versions, and execution
time. Only the manifest has a nondeterministic execution timestamp; metric
files are reproducible for unchanged inputs and configuration.

The analytics pipeline validates duplicate ticker/date keys, return bounds,
date order, and Silver adjusted-return provenance before calculation. It
reports unavailable ratios for zero volatility and insufficient observations.
It does not perform optimization, backtesting, or forecasting. Source data is
supplementary Yahoo Finance research data—not official, exchange-certified, or
proven free of survivorship bias. Adjusted prices do not independently
establish a complete corporate-action history.

## Phase 5: Statistical volatility forecasting

Phase 5 estimates **future 20-observed-session realized volatility** from the
Silver `daily_return` field. It does not predict stock prices or market
direction. Only existing Silver inputs are read; Silver checksums are checked
after processing, and raw-source hashes are compared with the Phase 3 manifest.
Raw, Silver, and Phase 4 outputs are not changed.

Install dependencies into the project environment and run from the project
directory:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.forecasting.cli --input-dir data/processed/silver --output-dir data/processed/forecasting/statistical --raw-dir data/raw/nse
```

The horizon, EWMA decay, date boundaries, and GARCH minimum history can be
configured:

```powershell
.\.venv\Scripts\python.exe -m src.forecasting.cli --horizon 20 --ewma-lambda 0.94 --train-end 2022-12-31 --validation-end 2023-12-31 --min-garch-observations 252
```

### Target and models

For origin `t`, the target uses only the next `H=20` observed Silver rows for
that ticker:

```text
RV(t, H) = sqrt((252 / H) * sum(r[t+1]^2, ..., r[t+H]^2))
```

This is annualized zero-mean realized volatility, not price change or
direction. Annualization uses 252 sessions per year. An incomplete horizon or
any missing/invalid return makes the target unavailable; no values are filled.
The target follows observed per-ticker rows and does not infer a complete NSE
holiday calendar.

The models use returns through each forecast origin and compare:

1. **Historical rolling baseline:** annualized sample standard deviation
   (`ddof=1`) of the trailing 20 returns through `t`, used as a persistence
   forecast. This backward-looking estimate is not itself a forward
   calculation.
2. **EWMA:** `v[t] = lambda * v[t-1] + (1-lambda) * r[t]^2`, default
   `lambda=0.94`. Initialize with the sample variance (`ddof=1`) of the first
   20 contiguous returns. With no long-run mean reversion in this EWMA model,
   hold the current variance constant over future steps and report
   `sqrt(252 * v[t])`. Lambda is configurable, not universally optimal, and
   is not tuned on the test data.
3. **GARCH(1,1):** fit a constant-mean model with normal innovations using the
   `arch` package:
   `h[t] = omega + alpha * epsilon[t-1]^2 + beta * h[t-1]`. Returns are scaled
   by 100 for estimation; the package's default backcast initializes variance.
   The standard non-negative variance parameters and stationarity constraints
   are handled by the model fit. Forecast volatility is the square root of the
   average annualized conditional variance over the requested horizon. A
   minimum of 252 contiguous history observations is required by default.
   Convergence warnings and fit failures are recorded; a failed GARCH fit is
   never replaced by another model's forecast.

Missing or invalid returns are not repaired. They break contiguous model
history, and a forward target containing one is unavailable. Initial returns
without a prior adjusted close are explicitly reported. The Silver validation
report must confirm that `daily_return` was calculated from `adj_close`.

### Chronological evaluation

The default split is by forecast-origin date: training history through
2022-12-31, validation origins during 2023, and test origins during 2024.
Forecasts are emitted for validation and test origins only. Each model uses an
expanding history available through its origin; no future target observations
enter model fitting or forecast features. Validation targets crossing
2023-12-31 are marked and excluded from validation metrics. Test targets must
fit fully within the available 2024 data; final origins lacking 20 subsequent
observations remain visible with unavailable targets. Model choices are not
tuned on the final test period.

MAE and RMSE compare annualized volatility. QLIKE compares variance:

```text
QLIKE = x / f - ln(x / f) - 1
```

Here `x` and `f` are realized and forecast variance, each floored at
`1e-12` annualized-variance units to keep the metric finite for zero or
near-zero variance. Aggregate model/split and per-stock/model/split metrics
include valid counts, eligible origins, coverage, unavailable forecasts, and
failed-fit counts. Twenty-session targets overlap, so adjacent errors are
dependent. These metrics are descriptive, not independent-observation
significance tests; formal inference would require methods such as block
bootstrap or non-overlapping horizons.

### Outputs

```text
data/processed/forecasting/statistical/
  statistical_forecasts.csv
  statistical_model_metrics.csv
  statistical_stock_metrics.csv
  statistical_validation_report.json
  statistical_manifest.json
  plots/
    actual_vs_forecast_test.png
    model_comparison_test.png
    per_stock_performance_test.png
    forecast_errors_over_time_test.png
```

`statistical_forecasts.csv` is long-form, one row per origin, symbol, and
model. Fields include `date`, `symbol`, `split`, `target_end_date`,
`target_status`, `target_observations`, `actual_realized_volatility`, `model`,
`forecast_volatility`, `forecast_status`, contiguous
`history_observations`, `garch_convergence_flag`, `fit_warnings`, and
`forecast_error`. Failed forecast periods stay in the file with explicit
status and no fabricated value. The metric CSVs include `model`, `split`,
optional `symbol`, `eligible_origins`, `valid_forecasts`,
`forecast_coverage`, `failed_fit_count`, `unavailable_forecast_count`, `mae`,
`rmse`, and `qlike`.

The validation report documents input quality, exclusions, target and model
statuses, failed fits, splits, provenance, and checksum results. The
reproducibility manifest records Silver input hashes, raw checksum verification,
configuration, runtime versions, output checksums, and a processing timestamp.
For unchanged inputs and configuration, forecast and metric data are
deterministic; only the manifest timestamp varies. Plots show test-period
actuals, forecasts, errors, and comparisons; unavailable forecast values remain
gaps and are counted in legends and reports.

The dataset covers only five stocks from 2020-2024. It is not representative
of all Indian equities and is not claimed to be exchange-certified or free of
survivorship bias. Adjusted prices do not independently establish a complete
corporate-action history. Historical results do not establish future accuracy
or investment suitability.

## Phase 6: Machine-learning volatility forecasting

Phase 6 compares Random Forest, XGBoost, and Ridge regression with the Phase 5
historical rolling, EWMA, and GARCH forecasts. It predicts future 20-observed-
session realized volatility, not prices or market direction. It reads the
existing Silver dataset and Phase 5 forecasts; it does not download data or
modify either input. Ridge scaling is fit inside each training-only pipeline.
Feature construction is causal: daily/lagged returns, trailing return and
volatility statistics, adjusted-close moving-average distances, RSI, and
volume/turnover features use observations through the forecast origin only.

The target for origin `t` is the annualized zero-mean realized volatility
`sqrt((252 / 20) * sum(r[t+1]^2, ..., r[t+20]^2))`. Targets crossing the
training (`2022-12-31`) or validation (`2023-12-31`) boundary are unavailable
in that split. Models refit monthly by default and train only on labels whose
target end date is strictly before the fit date. Evaluation is chronological:
training history through 2022, validation origins in 2023, and test origins in
2024. The validation-period lowest-MAE ML model is reported separately; the
test period is not used to select it. Forecasts are aligned with Phase 5
targets and compared using MAE, RMSE, and variance-based QLIKE. Overlapping
20-session targets make adjacent errors dependent, so results are descriptive
and not independent-observation significance tests.

Run the ML comparison locally:

```powershell
.\.venv\Scripts\python.exe -m src.forecasting.ml.cli --input-dir data/processed/silver --statistical-forecasts data/processed/forecasting/statistical/statistical_forecasts.csv --output-dir data/processed/forecasting/ml --model-dir models/volatility
```

Outputs include `ml_forecasts.csv`, `model_comparison.csv`, aggregate and
per-stock metric CSVs, feature importance, validation and reproducibility
manifests, test-period plots, and serialized model artifacts with feature
order, fit dates, parameters, versions, and SHA-256 checksums. The validation
report records forecast coverage and fitting failures explicitly. Failed
forecasts remain unavailable rather than being replaced with synthetic values.

## Phase 7: Constrained portfolio optimization and walk-forward backtesting

Phase 7 compares an equal-weight baseline with long-only minimum-variance,
maximum-Sharpe, and historical-CVaR portfolios. The walk-forward engine uses
trailing 60-, 120-, or 252-observation windows and rebalances on the first
observed session of each month or quarter from 2023 onward. Each estimation
window ends strictly before its rebalance date; weights apply from the
rebalance date through the day before the next rebalance. Expected returns are
the trailing arithmetic mean annualized by 252, and covariance is the trailing
sample covariance annualized by 252.

The optimization objectives are:

```text
Minimum variance: min(w' Σ w)
Maximum Sharpe:   max((w' μ - annual risk-free rate) / sqrt(w' Σ w))
CVaR:             min empirical tail loss at the configured confidence level
```

Weights sum to one and are bounded by the configured minimum and maximum
per-asset weights. An optional maximum-holdings constraint is enforced by
testing feasible asset subsets. The CVaR strategy uses only historical daily
return scenarios in the estimation window. The forecast-aware minimum-variance
variant uses successful ML volatility forecasts whose origin precedes the
rebalance date, scaled by a trailing, prior-only sample correlation matrix:
`Σ_forecast = D_forecast R_trailing D_forecast`. It is reported unavailable
when all required forecasts or correlations are not available; it does not
silently fall back to another covariance estimate.

Turnover is one-way: the initial investment is `sum(abs(w))` (equal to 1.0 for
a fully invested long-only portfolio) and later rebalances use
`0.5 * sum(abs(w_new - w_old))`. Configured basis-point costs apply at
rebalance and net returns are `(1 + gross_return) * (1 - cost_fraction) - 1`.
The default sensitivity grid covers 60/120/252-session windows, monthly and
quarterly rebalances, and 0/10 bps transaction costs. Metrics are reported
separately for validation (2023), test (2024), and the combined backtest, with
gross and net performance, turnover, costs, VaR/CVaR, and drawdown. Sensitivity
results are reported, not used to tune against the final test period.

Run the default local backtest:

```powershell
.\.venv\Scripts\python.exe -m src.optimization.cli --input-dir data/processed/silver --ml-comparison-csv data/processed/forecasting/ml/model_comparison.csv --output-dir data/processed/optimization --raw-dir data/raw/nse --windows 60 120 252 --frequencies monthly quarterly --maximum-weights 0.6 --transaction-cost-bps 0 10 --cvar-confidences 0.95 --rebalance-start 2023-01-01
```

The optimizer writes `optimized_weights.csv`, daily gross/net backtest returns,
strategy performance and risk metrics, rebalance turnover, solver diagnostics,
a validation report, plots, and a checksum/version manifest under
`data/processed/optimization/`. The report checks weight feasibility, prior-
only forecast origins and estimation windows, unchanged Silver/ML inputs, and
raw source checksums against the Phase 3 manifest. A solver failure or missing
forecast remains explicit in the diagnostics.

These are historical research results for five stocks from 2020-2024, not a
representative Indian-market sample or investment advice. The source is not
exchange-certified and may have survivorship bias. No risk-free series,
exchange calendar, or complete independent corporate-action history is
provided; the default annual risk-free rate is a configurable 0% computational
assumption. Transaction costs are modeled as a simple proportional charge and
do not include taxes, market impact, liquidity, slippage, or execution delays.
Forecasts, optimized weights, and historical backtests do not establish future
performance or investment suitability.

## Phase 8A: Optional S3 data lake integration

The S3 integration is opt-in. Importing modules, displaying validated
configuration, listing upload plans, and the dry-run command do not create an
AWS client or contact AWS. Upload, bucket-check, remote-list, download, and
verification commands contact AWS only when explicitly invoked. No bucket is
created automatically. The configured bucket name is supplied at runtime and
is not fixed in source code.

### Local-to-S3 mapping

| Local directory | S3 prefix |
|---|---|
| `data/raw/nse/` | `bronze/nse/` |
| `data/processed/silver/` | `silver/` |
| `data/processed/analytics/` | `gold/risk-analytics/` |
| `data/processed/forecasting/statistical/` | `gold/statistical-forecasts/` |
| `data/processed/forecasting/ml/` | `gold/ml-forecasts/` |
| `data/processed/optimization/` | `gold/portfolio-optimization/` |
| `models/volatility/` | `models/volatility/` |

Relative paths and filenames are preserved. No other project directory is
scanned. `.git`, virtual environments, caches, symlinks, credentials, common
secret/key files, and temporary files are excluded. Planning calculates local
file hashes and sizes; it does not transmit file contents.

### Configuration and setup

Install dependencies into the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Set the target configuration in the PowerShell session. Use an existing bucket
name that you are authorized to access; the example is a placeholder, not a
bucket created by FinCloud.

```powershell
$env:FINCLOUD_S3_BUCKET = "your-existing-bucket-name"
$env:FINCLOUD_AWS_REGION = "ap-south-1"
$env:FINCLOUD_S3_DRY_RUN = "true"
```

`FINCLOUD_S3_BUCKET` is required. `FINCLOUD_AWS_REGION` defaults to
`ap-south-1` and `AWS_REGION` is accepted as a fallback. The optional
`FINCLOUD_S3_DRY_RUN` setting accepts `true`/`false` values. Configuration
contains no credential fields. When a remote command is explicitly run,
boto3 uses its standard credential provider chain (for example an already
configured local AWS profile or an approved runtime role). Do not put keys in
source code, `.env` files, command history, or manifests; FinCloud does not
request or persist AWS credentials.

### Local-only preview commands

These commands do not call AWS. The default CLI invocation only prints help.
`plan` lists eligible filenames, sizes, and local SHA-256 values.
`dry-run` also writes a timestamped JSON plan manifest without overwriting an
existing manifest.

```powershell
.\.venv\Scripts\python.exe -m src.storage.s3_cli
.\.venv\Scripts\python.exe -m src.storage.s3_cli config
.\.venv\Scripts\python.exe -m src.storage.s3_cli plan
.\.venv\Scripts\python.exe -m src.storage.s3_cli dry-run
```

The dry-run manifest is written under
`data/processed/s3_manifests/`. Pass `--manifest <new-path>` to choose another
location; manifest creation is exclusive and refuses to replace an existing
file. Example output:

```text
Eligible files: 42
Missing mapped directories: 0
data/raw/nse/RELIANCE.csv -> bronze/nse/RELIANCE.csv (142227 bytes, sha256=<local digest>)
...
Local manifest: <project>\data\processed\s3_manifests\upload-<UTC timestamp>.json
Dry run complete; no AWS client was created and no data was uploaded.
```

Counts and hashes depend on the files present locally; the example is
illustrative, not a claim about a new run.

### Commands that contact AWS

Only run these after reviewing a dry-run plan, configuring an existing bucket,
and arranging access independently. `check` is read-only: it calls STS
`GetCallerIdentity`, S3 `HeadBucket`, and S3 `GetBucketLocation`; it prints the
AWS account and ARN, whether the bucket is accessible, and whether its actual
region matches the configured region. It does not create, upload, or alter
anything. The identity ARN is caller identity information, not a credential.

`upload` requires the explicit `--confirm` flag. It runs the same identity,
bucket-access, and region preflight before uploading. Each object is sent with
S3's conditional `If-None-Match: *` write, so an existing key (including one
created concurrently) is not overwritten. Conditional single-request uploads
are limited to 5 GB per file; larger files are rejected. The uploader uses
SSE-S3 (`AES256`), sets no ACL, hashes each local file, and verifies the
uploaded bytes by streaming the object back and comparing SHA-256. A local
manifest records planned and completed files, timestamps, hashes, verification
results, and any failures. A pre-existing remote key or an individual failure
causes the upload to stop for that key and be recorded in the manifest; do not
assume all files succeeded unless the final status is completed and all checks
passed.

```powershell
$env:FINCLOUD_S3_DRY_RUN = "false"
.\.venv\Scripts\python.exe -m src.storage.s3_cli upload --confirm

.\.venv\Scripts\python.exe -m src.storage.s3_cli check
.\.venv\Scripts\python.exe -m src.storage.s3_cli list --prefix gold/
.\.venv\Scripts\python.exe -m src.storage.s3_cli verify --key silver/all_stocks.csv --file data/processed/silver/all_stocks.csv
.\.venv\Scripts\python.exe -m src.storage.s3_cli download --key silver/all_stocks.csv --destination .\downloaded-all-stocks.csv --confirm
```

For Git Bash, set the same configuration and run commands in this order. The
`check` command and the unit tests are read-only/offline respectively; the
`upload` command is the step that writes objects to the existing bucket.

```bash
set -e
export FINCLOUD_S3_BUCKET="fincloud-market-data-unique123"
export FINCLOUD_AWS_REGION="ap-south-1"
export FINCLOUD_S3_DRY_RUN="false"

./.venv/Scripts/python.exe -m src.storage.s3_cli plan
./.venv/Scripts/python.exe -m unittest discover -s tests -v
./.venv/Scripts/python.exe -m src.storage.s3_cli check
./.venv/Scripts/python.exe -m src.storage.s3_cli upload --confirm
```

The check requires AWS credentials supplied through boto3's standard provider
chain and permissions for STS `GetCallerIdentity`, S3 `HeadBucket`, and
`GetBucketLocation`. Upload additionally needs permission to write objects and
read them back for checksum verification.
The upload command contacts AWS and writes to the bucket; run it only after
reviewing the local plan and a successful `check`. `set -e` makes this sequence
stop if a local check, test, or the AWS connection check fails.

`upload --dry-run` is an additional local-only preview form. Download refuses
to overwrite an existing destination. Missing objects and access/network
failures are surfaced. S3 ETags are metadata only and are never treated as
SHA-256 checksums. Remote verification computes SHA-256 from the actual
downloaded object stream; upload metadata alone is not considered proof of
remote integrity.

### AWS Console prerequisites and security

Before any contact command, manually confirm in AWS Console that the intended
bucket already exists in `ap-south-1` and that the bucket owner and authorized
operator are correct. Keep S3 Block Public Access enabled and do not add public
ACLs or bucket policies. Grant only the bucket/prefix permissions needed for
the chosen operation (bucket listing/access checks, object listing, object
read/head, and object write); do not grant bucket creation or deletion,
object deletion, or public access. Use an approved credential provider
configured outside this project. The integration does not create, configure,
or delete AWS resources.

S3 storage, requests, and streamed integrity verification can incur AWS
charges. No additional AWS services are used or provisioned. Phase 8A was
implemented and tested locally only; no AWS credentials were inspected and no
AWS API call, upload, download, or resource creation was performed.

## Project layout

- `src/ingestion/` — provider adapter, configuration, pipeline, and validation
- `src/preprocessing/` — local Silver-layer processing and manifest generation
- `src/analytics/` — stock performance, portfolio risk, correlation, beta, and reporting
- `src/forecasting/` — realized-volatility targets, statistical models, evaluation, reporting, and CLI
- `src/forecasting/ml/` — causal ML features, models, evaluation, artifacts, and CLI
- `src/optimization/` — constrained portfolio solvers, walk-forward engine, reporting, and CLI
- `src/storage/` — environment configuration, local S3 planning, client abstraction, manifests, and CLI
- `models/volatility/` — fitted Phase 6 model artifacts and metadata
- `data/raw/` — downloaded source data, including `data/raw/nse/`
- `data/processed/` — validation reports, Silver datasets, analytics, forecasts, and backtests
- `data/sample/` — small, non-sensitive sample datasets
- `notebooks/` — exploratory analysis notebooks
- `tests/` — automated offline tests
- `experiments/` — local prototypes and experiments
- `infrastructure/aws/` — reserved for future infrastructure work
- `research/` — project and domain research

All Phase 2 development and tests are local. AWS is not connected or
configured, and no AWS resources are created.

## Phase 9B: Local AWS-ready application

Phase 9B adds a React/Vite dashboard and an AWS Lambda-compatible Python API.
The local API uses an in-memory S3 mock seeded from project files; it does not
access AWS or claim cloud execution. Production code reads the Silver dataset
and verified Ridge artifacts from S3 and writes analysis results to the
existing bucket's Gold results prefix.

See [docs/PHASE_9B_LOCAL_AND_DEPLOYMENT.md](docs/PHASE_9B_LOCAL_AND_DEPLOYMENT.md)
for local run/test commands, API routes, required S3 object keys, dependency
and Docker validation, security/cost notes, and the approval-gated deployment
sequence. No AWS deployment or bucket changes have been performed.
