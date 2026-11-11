output "website_url" {
  description = "Public HTTP S3 website endpoint. S3 website endpoints do not provide HTTPS without a CDN or reverse proxy."
  value       = "http://${aws_s3_bucket_website_configuration.site.website_endpoint}"
}

output "bucket_name" {
  description = "Existing bucket hosting the root website objects."
  value       = data.aws_s3_bucket.existing.id
}

output "website_object_keys" {
  description = "Exact root-level object keys granted public GetObject access."
  value       = sort(local.website_files)
}

output "api_url" {
  description = "Invoke URL for the FinCloud HTTP API."
  value       = aws_apigatewayv2_api.api.api_endpoint
}

output "lambda_function_name" {
  description = "FinCloud Lambda function name."
  value       = aws_lambda_function.api.function_name
}
