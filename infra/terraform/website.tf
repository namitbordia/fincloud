resource "aws_s3_bucket_website_configuration" "site" {
  bucket = data.aws_s3_bucket.existing.id

  index_document {
    suffix = "index.html"
  }

  # S3 returns the app shell for unknown paths so the frontend can render its
  # client-side interface. The current UI uses in-app navigation rather than
  # URL-based routes, so this is a simple safe fallback.
  error_document {
    key = "index.html"
  }
}

resource "aws_s3_bucket_public_access_block" "site" {
  bucket = data.aws_s3_bucket.existing.id

  block_public_acls       = true
  ignore_public_acls      = true
  block_public_policy     = false
  restrict_public_buckets = false
}

resource "aws_s3_object" "site_files" {
  for_each = toset(local.website_files)

  bucket = data.aws_s3_bucket.existing.id
  key    = each.value
  source = "${local.website_directory}/${each.value}"

  source_hash            = filemd5("${local.website_directory}/${each.value}")
  content_type           = lookup(local.file_content_types, lower(element(reverse(split(".", each.value)), 0)), "application/octet-stream")
  server_side_encryption = "AES256"
  cache_control          = each.value == "index.html" ? "no-cache, no-store, must-revalidate" : "public, max-age=31536000, immutable"

  depends_on = [aws_s3_bucket_public_access_block.site]
}

data "aws_iam_policy_document" "public_site_objects" {
  statement {
    sid    = "PublicReadWebsiteObjectsOnly"
    effect = "Allow"

    principals {
      type        = "AWS"
      identifiers = ["*"]
    }

    actions = ["s3:GetObject"]
    resources = [
      for key in local.website_files : "${data.aws_s3_bucket.existing.arn}/${key}"
    ]
  }
}

resource "aws_s3_bucket_policy" "public_site_objects" {
  bucket = data.aws_s3_bucket.existing.id
  policy = data.aws_iam_policy_document.public_site_objects.json

  depends_on = [
    aws_s3_bucket_public_access_block.site,
    aws_s3_object.site_files,
  ]
}
