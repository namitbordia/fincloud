# FinCloud Phase 9: AWS Architecture and Deployment Plan

## Status and scope

This document records a local repository audit and a proposed AWS design. It
does not claim that FinCloud is deployed or that the configured S3 bucket,
objects, IAM policies, or AWS services were checked. No AWS credentials were
read, no AWS API calls were made, and no AWS resources or S3 objects were
created, changed, uploaded, or deleted.

The design uses the existing financial calculations and stored model artifacts.
It does not replace them with sample or fabricated results. The first cloud
vertical slice should use the existing five-stock Silver dataset, calculate
risk in Lambda, load a real saved Ridge model from S3 for inference, and run an
existing minimum-variance solver in Lambda.

## 1. Existing project audit

### Current application shape

FinCloud is currently a local-first Python research and command-line project.
There is no browser frontend, HTTP server, API route, Lambda handler, or
deployed infrastructure definition. `infrastructure/aws/` contains only a
placeholder. The top-level Python entry points are command-line modules:

| Area | Existing implementation and entry point | Current inputs/outputs |
|---|---|---|
| Market data ingestion | `src/ingestion/market_data.py`; `python -m src.ingestion.market_data` | Yahoo Finance via `yfinance`; writes local CSV and optional Parquet under `data/raw/nse/` and a validation report |
| Bronze-to-Silver preprocessing | `src/preprocessing/pipeline.py`; `python -m src.preprocessing.pipeline` | Reads local raw files; writes local Silver CSV/Parquet, quality report, and SHA-256 manifest |
| Risk and performance analytics | `src/analytics/`; report/CLI in `src/analytics/reporting.py` | Reads local Silver CSV; computes portfolio/stock metrics, returns, drawdown, VaR/CVaR, correlation, and optional benchmark beta; writes local tables, JSON, plots, and manifests |
| Statistical volatility forecasts | `src/forecasting/models.py`, targets/evaluation/reporting; `python -m src.forecasting.cli` | Rolling historical, EWMA, and fitted GARCH(1,1) forecasts against local Silver history; writes local forecast and evaluation artifacts |
| ML forecasts | `src/forecasting/ml/`; `python -m src.forecasting.ml.cli` | Builds causal features, trains/refits Ridge, Random Forest, and XGBoost models locally, evaluates them, serializes joblib artifacts, and writes local comparison outputs |
| Portfolio optimization/backtest | `src/optimization/`; `python -m src.optimization.cli` | Reads local Silver returns and ML forecast-comparison CSV; runs walk-forward scenarios and optimization; writes local weights, returns, risk metrics, diagnostics, plots, and manifests |
| S3 integration | `src/storage/`; `python -m src.storage.s3_cli` | Explicit opt-in local-to-S3 planning, uploads, downloads, listings, and checksum verification; credentials come from boto3's provider chain |
| Tests | `tests/` (14 test modules) | Offline unit tests; S3 tests use mocks and temporary local files |

`README.md` documents Phases 2-8A, the five-symbol historical data limitations,
CLI commands, local outputs, and optional S3 operations. `requirements.txt`
lists dependencies without version pins. There are no frontend package
manifests, notebooks, Lambda/SAM/CDK templates, or HTTP/API tests in the
audited project. `notebooks/`, `research/`, `experiments/`, and
`infrastructure/aws/` have no implemented app/runtime code.

### Reusable calculations and Lambda adaptation

| Component | Reuse in Lambda | Required adaptation |
|---|---|---|
| `src.analytics.portfolio`, `returns`, `performance`, `risk_metrics`, `correlation`, and `beta` | Core numeric functions are suitable to call on validated in-memory data. Sortino, historical VaR/CVaR, max drawdown, and correlation are implemented. | Add typed/validated HTTP request and response adapters, read Silver data from S3 rather than a local input path, and JSON-serialize unavailable/non-finite values as `null`. The analytics reporting CLI is file-oriented and should not be used as the HTTP handler. |
| `src.forecasting.models` | Historical rolling and EWMA forecasts are small in-memory calculations. GARCH is implemented with `arch`. | GARCH fits a model during each forecast and can use materially more CPU/time. Bound request history and measure Lambda duration; surface fit/convergence failures rather than inventing values. Reporting/plots are not request-path requirements. |
| `src.forecasting.ml.features`, `ml.models`, and `ml.evaluation` | Feature construction, estimator construction, inference, and metric calculation are reusable. Existing features include lagged/trailing returns, volatility, moving-average distances, RSI, and volume/turnover. | The current pipeline is a local training/backtest pipeline, not an inference service. Add a separate S3-backed artifact loader and request-time inference service; validate model name, symbol, feature order, artifact checksum, and input history. The `joblib` models are executable serialized artifacts, so only load artifacts controlled by the project. |
| `src.optimization.solvers` | Existing SciPy solvers implement minimum variance, maximum Sharpe, and CVaR minimization. Equal weight is implemented in analytics/backtest helpers. | Add a bounded request-time adapter that builds the selected return matrix, expected returns, covariance, and optional forecast covariance from S3 data. The current CLI/reporting path runs local CSV-based walk-forward/sensitivity backtests and plots; it is not itself a stateless API handler. Do not silently substitute another strategy when a solve is infeasible. |
| `src.ingestion` and `src.preprocessing` | Useful for controlled offline preparation or a later separately designed data refresh job. | Do not run Yahoo Finance ingestion for browser requests. Ingestion/preprocessing write local paths and require separate scheduling, data-source/license decisions, and cloud job design. The query runtime should use the existing S3 Silver dataset. |
| `src.storage.s3_client` and `s3_config` | Reuse configuration validation, boto3 default credential resolution, explicit errors, and checksum concepts. | Existing `download_file` writes to a local destination and the CLI is local-file oriented. Add an in-memory S3 object read (or a small Lambda adapter) for `pandas.read_csv`; do not rely on project files or a local fallback. The API role should be read-only for the dataset and model keys. |
| CLI/reporting modules | Reuse validation and domain functions selectively. | `argparse`, `Path`-based local inputs/outputs, plots, output directories, and whole-run manifests need a request/response service layer. Avoid generating plots or full historical backtests synchronously in the first endpoint. |

The user's current requirements cover a five-symbol universe, not every NSE
listing. The locally recorded Silver manifest reports five symbols and dates
from 2020-01-01 through 2024-12-31, 1,238 rows per symbol (6,190 combined), and
no excluded source rows. This is evidence from a local manifest, not proof of
the current contents of S3. The data source is Yahoo Finance via `yfinance`;
the README correctly notes that it is supplementary, not official or
exchange-certified data, with survivorship/corporate-action limitations.

### S3 status and current data access

The existing S3 wrapper has upload, download-to-file, list, object metadata,
existence, and checksum-verification operations. The manifest mapping sends
raw data to `bronze/nse/`, Silver to `silver/`, and analytics/forecast/
optimization outputs under sub-prefixes of `gold/`; model files go under
`models/volatility/`. The current mapper does not use the existing
`experiments/` or `metadata/` prefixes.

Two local manifests are present. Both have `status: dry-run`, list 83 planned
files, have no upload timestamps, and mark no remote checksums verified. One
uses an example bucket name and the other records the user's bucket name.
Neither establishes that any object is currently in S3. The actual bucket
region, access, encryption settings, public-access settings, and objects remain
unverified by this audit.

The API must use actual S3 `GetObject` results for data and model files. S3
access errors, missing keys, invalid manifests, and checksum/schema mismatches
must be surfaced as failures. No `data/`, `models/`, or other project-local
fallback is allowed in the deployed runtime. The browser must never receive
S3 write permissions or credentials.

## 2. Recommended AWS application architecture

### Recommended design

Use a private static frontend and a request-driven Python backend in
`ap-south-1`:

```text
Professor's browser
  ├─ HTTPS -> CloudFront -> separate private frontend S3 bucket (OAC)
  └─ HTTPS -> API Gateway HTTP API -> Python Lambda container image (ECR)
                                      ├─ read-only GetObject -> existing FinCloud S3 data lake
                                      │    ├─ silver/all_stocks.csv (+ Silver manifest)
                                      │    └─ models/volatility/* (+ model metadata)
                                      └─ CloudWatch Logs
                 Cognito User Pool -> HTTP API JWT authorizer (recommended)
```

Keep the existing data lake private and unchanged. Put the static site in a
separate frontend bucket and use CloudFront Origin Access Control (OAC) rather
than a public S3 website bucket. Build a static React/Vite frontend because no
frontend currently exists; host its generated assets on that private origin.
No custom domain, Route 53, NAT Gateway, database, EC2, RDS, or SageMaker
endpoint is required for the proposed read-only, five-symbol application.

Use one API Gateway **HTTP API** front door for the browser routes and one
Python Lambda initially. HTTP API provides route integration and stage
throttling; configure CORS to the exact deployed frontend origin. Add a
Cognito User Pool/JWT authorizer for a protected professor/demo account, so
the API does not become an unmetered anonymous compute endpoint. A Function
URL is a valid simpler alternative for a single handler and supports HTTPS and
CORS, but it does not provide API management, per-client throttling, request
validation, or WAF. `AWS_IAM` Function URL auth is not directly usable by a
public browser without signing requests with AWS credentials; `NONE` makes the
URL publicly invocable and should be a deliberate, explicitly approved choice.
Current Lambda guidance also requires both `lambda:InvokeFunctionUrl` and
`lambda:InvokeFunction` permissions for Function URL invocation.
The deployment decision is not made against AWS in this audit.

The Lambda execution role should have only `s3:GetObject` on the exact required
Silver and model object paths, and no `PutObject`, `DeleteObject`, bucket
creation, or broad `ListBucket` permission unless a verified design requires
it. Resolve the bucket name and region from deployment configuration, never
from browser input. Use the Lambda role, not embedded credentials. Keep S3
Block Public Access enabled, require TLS, use the bucket's approved encryption
at rest, and do not change the existing bucket configuration without separate
approval.

Use the Lambda container-image format for the full SciPy/scikit-learn/XGBoost
runtime unless a measured Linux ZIP build proves it fits the ZIP quota. Store
the image in Amazon ECR. Build for the matching Lambda OS/runtime and CPU
architecture; the Windows `.venv` contents cannot be copied into Lambda.
Omit ingestion-only, plotting-only, and optional Parquet dependencies from
the inference image if the runtime uses CSV. Use a configured Python runtime
supported by Lambda at deployment time, initially aligned with the project's
Python 3.11 baseline.

### Runtime API and screen data

Suggested endpoints (all results are computed/read by the backend):

| Endpoint | Data/action |
|---|---|
| `GET /api/health` | Report the invoked Lambda region/function metadata and verify the configured Silver object can actually be read from S3. Never return an `ok` cloud status based only on frontend constants. |
| `GET /api/universe` | Read S3 Silver data/manifest and return available symbols, date coverage, row counts, and supported forecast/optimization methods. |
| `POST /api/risk` | Validate symbols, period, and exact 100% weights; read historical returns from S3 and call the existing analytics functions. Return portfolio return, CAGR, volatility, Sharpe, Sortino, max drawdown, VaR/CVaR, and aligned correlation/observation counts. |
| `POST /api/forecast` | Validate symbol/method; load the appropriate existing model from S3 or call the statistical forecasting code in Lambda; return actual model status, forecast definition, history window, and available evaluation metrics. |
| `POST /api/optimize` | Validate assets/window/constraints; call existing solver logic using S3 returns and, for forecast-aware optimization, a real prior-origin forecast. Return weights, objective/solver diagnostics, expected return, volatility, Sharpe, drawdown, and explicit unavailable/failure statuses. |

The frontend should render the requested six screens from these actual
responses. The dashboard universe/coverage is S3-derived. The Portfolio Builder
can validate weights for usability in the browser, but the backend must repeat
all validation. Correlation must preserve missing values/pairwise counts;
never fill missing returns with zero. Treat `null`/unavailable metrics as
unavailable, not zero.

The Cloud Architecture screen should distinguish declared configuration from
verified execution. Report `Amazon S3`, `AWS Lambda`, region, and model source
only when the corresponding backend request confirms the Lambda invocation and
S3 read. A static frontend can say which architecture is configured, but must
not present it as a live status before verification. Do not expose internal
ARNs, account IDs, object contents, or credentials in the UI.

For the first version, keep browser state in memory and make the API
read-only. If result persistence is later required, add a separate,
explicitly scoped write path (for example `gold/api-results/`), a retention
policy, and a request identifier. Do not grant the query Lambda general S3
write access just to store optional reports.

## 3. ML and optimization deployment feasibility

### ML inference is not model training

There are local joblib artifacts for Ridge, Random Forest, and XGBoost for the
five configured symbols (15 artifacts total, about 0.92 MB across the local
model directory). `model_metadata.json` records training/evaluation
information, feature order, versions, and artifact hashes. These are currently
local files. No Lambda model loader or S3 inference route exists.

The simplest real cloud inference path is to upload the selected, trusted
artifacts and matching metadata to the existing `models/volatility/` prefix
after deployment approval; the Lambda loads the artifact from S3 and runs
`predict()` using the existing canonical feature builder. Begin with Ridge
inference because its artifacts are small; include its scikit-learn runtime.
Random Forest and XGBoost can follow once the Linux image is built and tested.
Do not claim retraining in AWS: existing ML training/refitting runs in the
local CLI. AWS training is only true after a future AWS compute invocation
actually fits a model and stores the resulting artifact/metadata in S3.

The models forecast future 20-observed-session realized volatility, not price
or market direction. Existing validation data can support historical
actual-versus-predicted plots and MAE/RMSE/QLIKE evaluation. For a new
forecast, realized future volatility and its error metrics do not exist yet;
show them as pending/unavailable until the target window is observed. Do not
relabel precomputed local evaluation rows as live AWS training or evaluation.

EWMA and historical rolling forecasts are directly suitable for request-time
calculation. GARCH uses `arch` to fit at runtime and reports convergence and
fit failures; test latency and cap history before placing it behind a
synchronous API. The existing optimizer consumes historical ML comparison
files for walk-forward forecast-aware covariance. Interactive forecast-aware
optimization needs an adapter that computes a valid forecast from the current
S3-backed history and applies the same prior-information rule. It must not
silently use future data or fall back when a forecast is missing.

### Dependency and Lambda package constraints

`requirements.txt` currently names `yfinance`, `pandas`, `numpy`, `pyarrow`,
`pytest`, `arch`, `matplotlib`, `scikit-learn`, `xgboost`, `joblib`, `scipy`,
and `boto3`, with no pins or deployment/dev split. Installed local versions
include Python 3.11.5, NumPy 2.4.6, pandas 3.1.1, SciPy 1.17.1,
scikit-learn 1.9.1, XGBoost 3.2.0, arch 8.0.0, PyArrow 26.0.1, and boto3
1.43.126. These are observations from the Windows virtual environment, not
AWS compatibility certification.

Approximate installed distribution-file sizes measured locally:

| Dependency | Local installed bytes | Runtime use |
|---|---:|---|
| NumPy | 56,078,126 | Shared numeric foundation |
| pandas | 76,328,980 | Data loading/analytics/features |
| SciPy | 138,781,615 | Portfolio optimization, GARCH dependencies |
| scikit-learn | 44,259,630 | Ridge and Random Forest |
| XGBoost | 145,533,891 | XGBoost inference |
| arch | 4,511,381 | GARCH |
| PyArrow | 93,752,406 | Optional Parquet I/O; avoid in a CSV-only handler |
| matplotlib | 34,704,844 | Local report plots; avoid in an API handler |
| yfinance | 1,360,243 | Local data ingestion only; avoid in request runtime |
| boto3 | 1,333,974 | S3 access |

These are Windows installed files, not the size of a built Linux ZIP/container;
dependency overlap and compression mean they are not an exact deployment
estimate. Nonetheless, the full scientific/ML set is clearly heavyweight.
Lambda ZIP deployments have a 50 MB direct-upload limit and a 250 MB
uncompressed package limit including layers; layers do not bypass the combined
limit. Lambda container images support substantially larger packages (up to
10 GB uncompressed). Therefore use a Linux container image for the full
SciPy/sklearn/XGBoost handler unless an actual target-platform ZIP build,
measured with all layers, proves it fits. The app's local Windows packages
cannot be used as-is. Exact Linux image size, cold-start time, memory, and
timeout remain to be measured.

The local Silver directory is about 3.26 MB and local model directory about
0.92 MB, so the data/artifact payload itself is modest. Read S3 objects as
streams/in-memory CSV; do not include historical datasets or `.joblib` files
inside the Lambda deployment bundle. Use bounded result payloads and keep
synchronous API work within the HTTP API timeout; move longer work to a later
asynchronous design only if measurements show it is needed.

## 4. Security and cost

### Security controls

- Keep the data-lake bucket private and unchanged; use a separate frontend
  bucket behind CloudFront OAC.
- Use IAM roles for Lambda and CloudFront origin access. Scope the API role to
  read only the required bucket keys. No browser S3 credentials, write access,
  embedded keys, root keys, or local-file fallback.
- Protect the browser API with Cognito JWT authorization by default. If
  anonymous access is desired instead, review and explicitly approve that
  public exposure before deployment; CORS is not authentication.
- Allow only the deployed frontend origin in CORS. Validate HTTP methods,
  symbols, date windows, body sizes, weights, and optimization bounds in the
  backend. Use API stage throttles and avoid logging submitted portfolios or
  full market-data payloads.
- Use HTTPS in transit and encryption at rest for S3/ECR/logs. Use an
  existing approved KMS key only if required; key requests can add cost.
  Configure CloudWatch log retention and alarms intentionally.
- Do not return secrets, credentials, full exception traces, or sensitive AWS
  identity details to the browser. Emit explicit safe errors and correlation
  IDs instead.

### Cost risks (not a price quote)

₹0 cannot be promised. No usage assumptions, billing history, region-specific
calculator inputs, or free-tier eligibility were verified. Check current
service pricing and the account's own Free Tier/credits before approval.
Potential charges include:

- S3 object storage, GET/PUT/LIST requests, and data transfer; object uploads
  or changes may incur request charges.
- Lambda request and duration charges, with the larger memory/CPU allocation
  potentially needed by SciPy/XGBoost/GARCH.
- ECR image storage and image transfer/pulls.
- API Gateway HTTP API requests and outbound data.
- CloudFront requests and data transfer; optional WAF/custom domain/DNS add
  separate cost.
- CloudWatch Logs ingestion/storage and any alarms/metrics.
- Cognito usage if used, and optional KMS key/request charges.

There is no always-running compute in the recommendation; Lambda is
request-driven. Avoid NAT Gateway, EC2, RDS, SageMaker endpoints, and
provisioned/warm compute for this demo. Keep the image and logs bounded,
configure budgets/alerts if available, and remove only newly created resources
after an approved teardown plan. Do not delete or alter existing bucket data.

## 5. Phased implementation and deployment plan

| Stage | Work and exit criteria |
|---|---|
| **9A: Project audit and architecture design** | Completed by this audit. Repository structure, reusable logic, gaps, AWS plan, sizing evidence, security, cost risks, and blockers are documented. |
| **9B: Refactor backend for cloud execution** | Add a typed request/service layer and Lambda handler. Replace local input paths with S3 reads; separate ML inference from training; validate requests and JSON responses; preserve all existing calculations; no local fallback. Add offline tests for routes, bad data, missing S3 keys, and calculation parity. |
| **9C: Prepare Lambda package and dependencies** | Add pinned/compatible runtime dependencies and a reproducible Linux container build (or prove a ZIP build fits). Omit yfinance, matplotlib, pyarrow, and pytest from the request image unless a documented runtime need appears. Test imports and artifacts in the target Lambda-compatible image and measure cold start, memory, and endpoint duration. |
| **9D: Prepare frontend and API integration** | Create the React/Vite browser UI and six screens. Add endpoint client, exact-origin CORS handling, Cognito sign-in, loading/error/empty states, and a visible cloud execution indicator driven only by backend responses. |
| **9E: Deploy backend and connect existing S3** | **Approval required before any AWS operation.** Review the bucket/prefix objects and region in Console, confirm permissions and cost, deploy SAM/CloudFormation resources in `ap-south-1`, grant read-only Lambda access, and upload only approved data/model objects if they are missing. No bucket policy or existing contents are changed without explicit approval. |
| **9F: Deploy frontend** | **Approval required before any AWS operation.** Build static assets, create/use a separate private frontend origin and CloudFront OAC, configure the API URL/Cognito client, set exact CORS origin, and publish the site without making the data lake public. |
| **9G: End-to-end cloud testing** | **Approval required for AWS calls.** From a browser, verify the live Lambda health/S3 read, universe/coverage, portfolio validation, risk calculations, Ridge inference from S3, minimum-variance optimization, response/error behavior, and no browser-to-S3 writes. Compare backend outputs to offline regression fixtures from existing real calculations. |
| **9H: Professor demonstration preparation** | Prepare a demo portfolio, explain source limitations and forecast horizon, verify sign-in and browser/network connectivity, record no unverified service status, and retain a short fallback plan using stored historical reports without presenting it as live cloud computation. |

### Vertical-slice order

1. Deploy a static frontend and health/universe API; verify actual Lambda
   invocation and S3 object reads.
2. Select stocks and submit weights; compute actual risk metrics in Lambda.
3. Load a real Ridge artifact from S3 and execute one-asset inference in
   Lambda. Show historical evaluation only where matching actual targets exist.
4. Run minimum variance using real S3-backed returns and the existing SciPy
   solver.
5. Add EWMA/GARCH, Random Forest/XGBoost, CVaR and maximum-Sharpe, then the
   forecast-aware strategy after prior-origin tests pass.
6. Add report persistence only if needed and separately approved.

## 6. Exact next commands and approval boundary

The repository is Windows-based. The current Python environment is already
installed. These commands are local-only; the test command was run during this
audit and passed:

```powershell
Set-Location 'C:\Users\Namit Bordia\Downloads\fincloud'
.\.venv\Scripts\python.exe -m pytest -q
node --version
npm --version
docker version
sam --version
```

The tool-version checks identify whether the next local frontend and
container/SAM build steps are available; they do not call AWS. Once the SAM
template and source changes exist, local-only validation/build commands will
be:

```powershell
sam validate --lint
sam build --use-container
```

No deployment command was run. After the code, cost estimate, account/bucket
preflight, and demo access choice are reviewed, a future deployment operation
would require a separate explicit approval before running any command such as
`sam deploy --guided --region ap-south-1` or any S3/CloudFront publish command.
These commands have not been run and must not be run until that approval.

## 7. Blockers and decisions before deployment

1. The actual S3 bucket objects, region, access, encryption, and policy are
   unknown; local dry-run manifests are not cloud verification.
2. The browser access mode needs confirmation before deployment. This design
   recommends Cognito sign-in; a public unauthenticated API is a deliberate
   cost/security tradeoff, not the default.
3. The ML artifacts are local. Confirm their checksums/versions and approve
   their S3 upload only if the expected keys are absent.
4. The Lambda-compatible Linux image has not been built or measured. Confirm
   target architecture, versions, image size, cold start, memory, and API
   latency with local container tests.
5. The configured stock universe is five stocks through 2024. The professor
   should be told this is a historical research sample, not the full Indian
   market or investment advice.
6. Cost depends on account-specific Free Tier/credits and actual traffic,
   storage, transfer, logs, and image size. Review the AWS Pricing Calculator
   before any deploy; no ₹0 guarantee is made.

## 8. AWS references

- [Lambda quotas and deployment limits](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html)
- [Python Lambda deployment packages](https://docs.aws.amazon.com/lambda/latest/dg/python-package.html)
- [Lambda Function URL access control](https://docs.aws.amazon.com/lambda/latest/dg/urls-auth.html)
- [Lambda pricing](https://aws.amazon.com/lambda/pricing/)
- [API Gateway pricing](https://aws.amazon.com/api-gateway/pricing/)
- [Amazon S3 pricing](https://aws.amazon.com/s3/pricing/)
- [CloudFront pricing](https://aws.amazon.com/cloudfront/pricing/)
- [Static S3 website hosting guidance](https://docs.aws.amazon.com/AmazonS3/latest/userguide/WebsiteHosting.html)
- [SAM deployment resources and build guidance](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/what-is-sam.html)
