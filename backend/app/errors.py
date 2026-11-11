"""Expected application errors mapped to safe HTTP responses."""


class RequestValidationError(ValueError):
    """The request is syntactically valid but violates the API contract."""


class S3ObjectNotFoundError(RuntimeError):
    """A required S3 object does not exist."""


class S3RepositoryError(RuntimeError):
    """An S3 operation failed or returned invalid content."""


class ForecastUnavailableError(RuntimeError):
    """A forecast cannot be produced from the supplied data or artifact."""


class OptimizationUnavailableError(RuntimeError):
    """No requested optimization strategy produced a valid solution."""
