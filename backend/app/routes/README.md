# FinCloud API routes

The deployed API Gateway HTTP API invokes the Python 3.12 Lambda handler in `backend/app/handler.py`. All operational market data, research summaries, and model artifacts are read from private S3 objects; local files are not used by the AWS handler. Analytics outputs are saved under `gold/api-results/`.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/health` | Lambda, region, API Gateway, and S3 connectivity status |
| GET | `/stocks` | Supported symbols and latest historical observations |
| GET | `/stocks/{symbol}/history` | Historical prices and returns |
| POST | `/portfolio/analyze` | Portfolio risk and correlation metrics |
| POST | `/portfolio/correlation` | Pairwise return correlations |
| POST | `/ml/forecast` | Statistical or trained Ridge/Random Forest volatility forecast |
| POST | `/portfolio/optimize` | Minimum variance, maximum Sharpe, CVaR minimization, or comparison |
| GET | `/research/experiments` | Curated experiment and validation summary |
| GET | `/research/models` | Trained artifact registry and metadata |

The API is publicly callable without end-user authentication. API Gateway throttling limits request rate; CORS permits the configured S3 website origin. These controls do not authenticate callers. The Lambda role grants `GetObject` only for its required S3 inputs and `PutObject` only to `gold/api-results/*`; it cannot list or delete bucket objects.

Local tests use mocked S3 clients. The development server is local-only and does not call AWS. See `infra/terraform/README.md` for deployment and cleanup instructions.
