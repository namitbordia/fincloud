# FinCloud Phase 9B: Local Application and AWS Preparation

## Status

The full-stack vertical slice is implemented for local development and offline
testing. The local API uses an in-memory S3 mock seeded from the checked-in
research data and model artifacts. It does not create a boto3 S3 client, read
AWS credentials, or call AWS. No AWS resources or S3 objects have been changed.

The production Lambda repository has no local-file fallback: it reads the
Silver data and Ridge artifacts from S3, verifies their manifests and
checksums, and writes analysis outputs to S3. The application marks cloud
execution verified only when it is running in Lambda and its required data
reads have succeeded. The existing dry-run manifests do not prove the
corresponding objects are present in S3.

## Existing implementation reused

- `src/analytics/`: portfolio return, risk/performance, and correlation
  calculations.
- `src/forecasting/ml/features.py`: canonical ordered Ridge feature builder.
- `models/volatility/model_metadata.json` and the existing per-symbol Ridge
  joblib files: training metadata and fitted inference artifacts. Training
  does not run as part of an API request.
- `src/forecasting/models.py`: existing EWMA baseline.
- `src/optimization/solvers.py`: constrained minimum-variance,
  maximum-Sharpe, and related solvers.
- Existing Silver manifest and SHA-256 metadata: S3 data integrity checks.

## Local run and tests

Run both processes from the repository root in separate PowerShell terminals.
The backend and Vite proxy are bound to loopback addresses.

```powershell
.\.venv\Scripts\python.exe -m backend.dev_server
```

```powershell
Set-Location frontend
npm ci
npm run dev
```

Open the Vite URL printed in the terminal (normally
`http://127.0.0.1:5173`). Local mode is explicitly labelled as mock execution;
the browser does not receive AWS credentials.

Run offline validation and the production frontend build:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
Set-Location frontend
npm ci
npm test
npm run build
```

To run the local API against a Lambda-shaped HTTP API event without starting
the server, the offline tests call `backend.app.handler.handle_event` with a
mock repository.

## Implemented API and application

| Route | Behavior |
|---|---|
| `GET /health` | Runtime and region, S3 object reachability, and verified-cloud status |
| `GET /stocks` | Stock universe, coverage, row count, and Silver manifest verification |
| `POST /portfolio/risk` | Existing performance/risk metrics, VaR/CVaR, correlation, and S3 result write |
| `POST /forecast` | Verified S3 Ridge inference or existing EWMA forecast and S3 result write |
| `POST /portfolio/optimize` | Minimum variance, maximum Sharpe, or comparison through existing solvers and S3 result write |

The React application includes the dashboard, stock and weight editor with
100% validation, risk/correlation results, Ridge/EWMA forecast selection,
minimum-variance/maximum-Sharpe optimization, and a cloud-architecture status
view. Cloud indicators distinguish the mock from a verified Lambda/S3 request.
The frontend includes optional Cognito authorization-code PKCE configuration;
the local mock does not require Cognito.

HTTP API v1 and v2 Lambda proxy event shapes, base64 request bodies, 1 MB
application request-body limit, JSON response encoding, CORS, and explicit
error responses are handled by the Lambda entry point. The current local
2020–2024 fixture returned a 67,127-byte risk response, a 1,536-byte Ridge
forecast response, and a 2,209-byte optimization comparison response through
the mock API. These are local measurements, not cloud performance claims.
The HTTP API integration has a 30-second timeout and 10 MB payload limit; the
backend's 1 MB request limit is intentionally stricter
([HTTP API quotas](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-quotas.html)).

## Required existing S3 objects

| Object key | Requirement |
|---|---|
| `silver/all_stocks.csv` | Combined Silver OHLCV, adjusted close, and daily-return table |
| `silver/silver_dataset_manifest.json` | Must include `output_sha256.all_stocks.csv`, `stock_symbols`, and row count |
| `models/volatility/model_metadata.json` | Ridge artifact hash, ordered feature list, target, and training metadata |
| `models/volatility/{SYMBOL}_ridge.joblib` | Trained artifact registered in the metadata for each forecastable symbol |

Results are written under `gold/api-results/risk/`,
`gold/api-results/forecasts/`, and `gold/api-results/optimization/`. The
objects must be confirmed present before deployment; missing data or model
objects fail explicitly. The Lambda role policy in
`infrastructure/aws/lambda-execution-policy.json` scopes reads to these
prefixes/objects and writes to the API result prefix. It is an example policy,
not an applied IAM change.

## Dependency and container assessment

The pinned backend set is NumPy, pandas, SciPy, scikit-learn, joblib, `arch`,
and boto3. The initial Lambda image omits XGBoost, PyArrow, Matplotlib,
yfinance, and test-only packages. `arch` remains necessary because the reused
forecasting module imports it even when the API calls EWMA.

I verified binary Python 3.11 Linux x86-64 wheels for the pinned requirements
using the `manylinux_2_28_x86_64` platform tag. The full resolved wheel set,
including transitive dependencies, downloads to about 106 MB. This is a
compressed wheel total, not the installed or final container size. The Lambda
base image and installed native libraries increase the final image. The
container image size could not be measured because the local Docker daemon is
unavailable; run the documented `docker build` and `docker image inspect`
commands on a machine with Docker Desktop running before choosing memory and
timeout settings.

The Dockerfile targets the AWS Lambda Python 3.11 x86-64 base image and installs
`libgomp` for native scientific dependencies. It has not yet been built or
executed inside Docker here. Lambda cold-start time, peak memory, request
latency, and an appropriate function timeout therefore remain unmeasured.
Benchmark against the real S3 data and model artifacts in an approved test
deployment before setting production memory or timeout values. Keep each
synchronous request comfortably within the API Gateway HTTP API integration
timeout. Risk results contain daily return observations, so verify response
sizes for the full selected time range.

## AWS architecture and preparation

Use the existing private S3 bucket for the data lake, model artifacts, results,
and (if approved) a dedicated `frontend/` prefix for built static assets. Keep
S3 Block Public Access enabled. A CloudFront Origin Access Control can read
only `frontend/*`; the API role gets only the Silver/model reads and
`gold/api-results/*` writes. Do not grant the browser S3 access.

Use:

1. **Amazon S3**: existing bucket and data; no new bucket is assumed.
2. **Amazon ECR**: store the Lambda container image.
3. **AWS Lambda (Python 3.11 container image)**: API compute using its
   execution role.
4. **API Gateway HTTP API**: five Lambda proxy routes and a native Cognito
   JWT authorizer. HTTP API is sufficient for this request/response API; a
   Function URL is not needed in addition.
5. **Amazon Cognito User Pool**: browser sign-in with authorization code + PKCE;
   allow only the frontend URL as callback and sign-out URL, and use the app
   client ID as the HTTP API JWT audience.
6. **Amazon CloudFront**: serve the static frontend through OAC to the private
   S3 `frontend/` prefix. The frontend calls the regional API Gateway URL
   directly; set its `VITE_API_BASE_URL` to that URL and configure the Lambda
   CORS allowlist to the exact CloudFront origin.
7. **Amazon CloudWatch Logs**: Lambda log group and API access logs, with
   retention and encryption set deliberately.

The implementation does not need SageMaker, EC2, RDS, a NAT Gateway, or an
always-running service. API Gateway HTTP API is chosen over REST API because
the application needs proxy routes, CORS, and JWT authorization—not REST-only
features such as usage plans, API keys, built-in caching, request
transformation, or WAF association. See [API Gateway HTTP APIs](https://aws.amazon.com/api-gateway/pricing/)
and [Amazon S3 pricing](https://aws.amazon.com/s3/pricing/) before deployment.
Potential charges include ECR image storage/scanning, Lambda requests and
duration, API Gateway calls, S3 requests/storage/data transfer, CloudFront
requests and data transfer, Cognito usage, and CloudWatch logs/metrics. S3
CloudTrail data events and detailed logging can also add cost; enable them
deliberately after reviewing current pricing. No service is assumed to be free.

### Build and publish the Lambda image (future, only after approval)

Run from the repository root on a machine with Docker Desktop running and
authorized AWS CLI access. The root `.dockerignore` limits the build context
to `backend/` and `src/`, excluding local datasets, model files, and frontend
dependencies. First inspect whether the ECR repository exists;
do not create a duplicate.

```powershell
$Region = "ap-south-1"
$AccountId = aws sts get-caller-identity --query Account --output text
$Registry = "$AccountId.dkr.ecr.$Region.amazonaws.com"
$Image = "$Registry/fincloud-api:phase9b"

docker build --platform linux/amd64 -f backend/Dockerfile -t fincloud-api:phase9b .
docker image inspect fincloud-api:phase9b --format '{{.Size}}'
aws ecr describe-repositories --repository-names fincloud-api --region $Region
```

If the repository exists and the reviewed policy/costs are approved:

```powershell
aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $Registry
docker tag fincloud-api:phase9b $Image
docker push $Image
```

Creating the ECR repository, any missing IAM role, the Lambda, API Gateway,
Cognito, or CloudFront is a separate AWS change and is not included in these
local steps.

### API, IAM, frontend, and authorization wiring (future, only after approval)

1. Confirm required S3 keys exist and match their manifests using an
   authorized, read-only review. Do not upload or replace project data as part
   of this implementation.
2. Review and tailor `infrastructure/aws/lambda-execution-policy.json` to the
   exact Lambda name, region, account, bucket, and prefixes. Pre-create the
   `/aws/lambda/fincloud-api` log group if retaining the current policy without
   `logs:CreateLogGroup`; choose log retention and encryption. Attach only
   `s3:GetObject` for the required Silver/model objects, `s3:PutObject` for
   `gold/api-results/*`, and scoped log writes. Never grant bucket public
   access, object deletion, browser writes, or broad bucket listing.
3. Create/update the Lambda from the ECR image with environment variables
   `FINCLOUD_S3_BUCKET`, `FINCLOUD_AWS_REGION`,
   `FINCLOUD_CORS_ORIGINS`, `FINCLOUD_SILVER_KEY`,
   `FINCLOUD_SILVER_MANIFEST_KEY`, `FINCLOUD_MODEL_PREFIX`, and
   `FINCLOUD_RESULTS_PREFIX`. Confirm runtime role and memory/timeout through
   measured testing; do not treat unmeasured values as production-ready.
4. Grant API Gateway permission to invoke only this Lambda, restricted by the
   API's execute-api source ARN and source account; do not leave a public
   Lambda resource permission.
5. Create an API Gateway HTTP API with Lambda proxy payload format 2.0 and
   routes `GET /health`, `GET /stocks`, `POST /portfolio/risk`,
   `POST /forecast`, and `POST /portfolio/optimize`. Add a JWT authorizer with
   issuer
   `https://cognito-idp.ap-south-1.amazonaws.com/<user-pool-id>` and audience
   `<app-client-id>`; protect all application routes. Configure only the exact
   frontend origin, `GET,POST,OPTIONS`, and `authorization,content-type`
   headers. Keep Lambda response CORS enabled for local and gateway proxy
   responses.
6. Configure a Cognito app client for authorization code grant and PKCE, with
   no client secret in the browser. Restrict callback and sign-out URLs to the
   final HTTPS CloudFront origin. Do not put tokens or credentials in logs or
   build-time source.
7. Run `npm ci` and `npm run build` in `frontend/` with
   `VITE_API_BASE_URL=https://<api-id>.execute-api.ap-south-1.amazonaws.com`,
   `VITE_COGNITO_DOMAIN`, `VITE_COGNITO_CLIENT_ID`, and
   `VITE_COGNITO_REDIRECT_URI` set to the approved values. Upload only the
   resulting `frontend/dist/` contents to the approved `frontend/` prefix;
   this would modify the existing bucket and is not authorized by this phase.
8. Create CloudFront with the existing S3 bucket REST origin and OAC, grant
   that distribution `s3:GetObject` only for `frontend/*` using a bucket
   policy condition scoped to the distribution ARN (and source account), and
   keep public access blocked. Do not use the S3 website endpoint or public bucket access.
   Verify callback URLs, CORS, JWT authorization, API results, and browser
   network requests before the demonstration.

S3 access logging, CloudTrail data events for the relevant prefixes, and
CloudWatch request/error metrics are useful for audit and troubleshooting;
scope them to the application, encrypt log destinations, set retention, and
review their cost before enabling. Use TLS in transit, S3 server-side
encryption, least-privilege IAM, and exact-origin CORS. No AWS setting has
been changed as part of Phase 9B preparation.

## Remaining blockers before a cloud demonstration

- Required data/model objects in S3 and their hashes have not been verified.
- Docker daemon unavailable here; final image build and size are unverified.
- Lambda cold-start, memory, timeout, and live S3 latency are unmeasured.
- The browser needs approved Cognito, API Gateway, CloudFront, and origin values.
- A private frontend prefix and its CloudFront OAC bucket-policy statement
  require explicit approval because they modify the existing bucket.
- The current deployment request forbids AWS changes. Deploy only after the
  owner explicitly approves each real AWS deployment operation.
