variable "aws_region" {
  description = "AWS region containing the existing FinCloud bucket."
  type        = string
  default     = "ap-south-1"
}

variable "bucket_name" {
  description = "Existing S3 bucket used for the FinCloud website. Terraform never creates or replaces this bucket."
  type        = string
  default     = "fincloud-market-data-unique123"
}

variable "site_directory" {
  description = "Path to the built Vite site. Defaults to the project's frontend/dist directory."
  type        = string
  default     = "../../frontend/dist"
}

variable "website_origin" {
  description = "HTTP origin allowed to call the public API from the static website."
  type        = string
  default     = "http://fincloud-market-data-unique123.s3-website.ap-south-1.amazonaws.com"
}

variable "lambda_package_path" {
  description = "Locally built Lambda zip package generated from backend/Dockerfile.lambda-package."
  type        = string
  default     = "../../backend/dist/fincloud-api.zip"
}
