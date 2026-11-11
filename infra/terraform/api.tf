data "aws_caller_identity" "current" {}

locals {
  lambda_package_path = abspath("${path.module}/${var.lambda_package_path}")
  lambda_package_hash = filebase64sha256(local.lambda_package_path)
  lambda_package_key  = "lambda/fincloud-api-${substr(local.lambda_package_hash, 0, 20)}.zip"
  api_routes = toset([
    "GET /health",
    "GET /stocks",
    "GET /stocks/{symbol}/history",
    "POST /portfolio/analyze",
    "POST /portfolio/correlation",
    "POST /ml/forecast",
    "POST /portfolio/optimize",
    "GET /research/experiments",
    "GET /research/models",
  ])
}

resource "aws_s3_object" "api_package" {
  bucket                 = data.aws_s3_bucket.existing.id
  key                    = local.lambda_package_key
  source                 = local.lambda_package_path
  source_hash            = local.lambda_package_hash
  content_type           = "application/zip"
  server_side_encryption = "AES256"
  cache_control          = "no-store"
}

resource "aws_iam_role" "api_lambda" {
  name = "fincloud-api-lambda-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "api_lambda" {
  name = "fincloud-api-data-access"
  role = aws_iam_role.api_lambda.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadRequiredPrivateDataAndModels"
        Effect = "Allow"
        Action = ["s3:GetObject"]
        Resource = [
          "${data.aws_s3_bucket.existing.arn}/silver/all_stocks.csv",
          "${data.aws_s3_bucket.existing.arn}/silver/silver_dataset_manifest.json",
          "${data.aws_s3_bucket.existing.arn}/models/volatility/model_metadata.json",
          "${data.aws_s3_bucket.existing.arn}/models/volatility/*_ridge.joblib",
          "${data.aws_s3_bucket.existing.arn}/models/volatility/*_random_forest.joblib",
          "${data.aws_s3_bucket.existing.arn}/experiments/ml/fincloud_research_summary.json",
        ]
      },
      {
        Sid      = "WriteApiResultsOnly"
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = "${data.aws_s3_bucket.existing.arn}/gold/api-results/*"
      },
      {
        Sid      = "WriteLambdaLogs"
        Effect   = "Allow"
        Action   = ["logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "${aws_cloudwatch_log_group.api.arn}:*"
      },
    ]
  })
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/fincloud-api"
  retention_in_days = 7
}

resource "aws_lambda_function" "api" {
  function_name    = "fincloud-api"
  description      = "FinCloud portfolio analytics and volatility inference API"
  role             = aws_iam_role.api_lambda.arn
  runtime          = "python3.12"
  handler          = "backend.app.handler.lambda_handler"
  architectures    = ["x86_64"]
  timeout          = 15
  memory_size      = 512
  s3_bucket        = data.aws_s3_bucket.existing.id
  s3_key           = aws_s3_object.api_package.key
  source_code_hash = local.lambda_package_hash

  environment {
    variables = {
      FINCLOUD_S3_BUCKET           = var.bucket_name
      FINCLOUD_AWS_REGION          = var.aws_region
      FINCLOUD_CORS_ORIGINS        = var.website_origin
      FINCLOUD_SILVER_KEY          = "silver/all_stocks.csv"
      FINCLOUD_SILVER_MANIFEST_KEY = "silver/silver_dataset_manifest.json"
      FINCLOUD_MODEL_PREFIX        = "models/volatility"
      FINCLOUD_RESULTS_PREFIX      = "gold/api-results"
      FINCLOUD_EXPERIMENTS_KEY     = "experiments/ml/fincloud_research_summary.json"
    }
  }

  depends_on = [aws_cloudwatch_log_group.api, aws_iam_role_policy.api_lambda]
}

resource "aws_apigatewayv2_api" "api" {
  name          = "fincloud-http-api"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = [var.website_origin]
    allow_methods = ["GET", "POST", "OPTIONS"]
    allow_headers = ["content-type"]
    max_age       = 300
  }
}

resource "aws_apigatewayv2_integration" "api" {
  api_id                 = aws_apigatewayv2_api.api.id
  integration_type       = "AWS_PROXY"
  integration_method     = "POST"
  integration_uri        = aws_lambda_function.api.invoke_arn
  payload_format_version = "2.0"
  timeout_milliseconds   = 15000
}

resource "aws_apigatewayv2_route" "api" {
  for_each  = local.api_routes
  api_id    = aws_apigatewayv2_api.api.id
  route_key = each.value
  target    = "integrations/${aws_apigatewayv2_integration.api.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.api.id
  name        = "$default"
  auto_deploy = true

  default_route_settings {
    throttling_burst_limit = 2
    throttling_rate_limit  = 0.3
  }
}

resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowFinCloudHttpApiInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api.execution_arn}/*/*"
}
