data "aws_s3_bucket" "existing" {
  bucket = var.bucket_name
}

locals {
  website_directory = abspath("${path.module}/${var.site_directory}")
  website_files     = fileset(local.website_directory, "**")
  file_content_types = {
    html  = "text/html; charset=utf-8"
    css   = "text/css; charset=utf-8"
    js    = "application/javascript"
    json  = "application/json"
    svg   = "image/svg+xml"
    png   = "image/png"
    jpg   = "image/jpeg"
    jpeg  = "image/jpeg"
    ico   = "image/x-icon"
    woff  = "font/woff"
    woff2 = "font/woff2"
  }
}
