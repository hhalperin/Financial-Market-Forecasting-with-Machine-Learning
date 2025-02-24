########################################################################
# s3.tf
########################################################################
resource "aws_s3_bucket" "website" {
  bucket        = var.domain_name
  force_destroy = true

  tags = {
    Name        = var.domain_name
    Environment = var.environment
  }
}

# Ownership controls for the bucket
resource "aws_s3_bucket_ownership_controls" "ownership" {
  bucket = aws_s3_bucket.website.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

# Public access is fully blocked; rely on CloudFront for distribution
resource "aws_s3_bucket_public_access_block" "block_public" {
  # Optionally ensure it's created after the bucket
  depends_on = [aws_s3_bucket.website]

  bucket                  = aws_s3_bucket.website.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Example: Upload local files only if enabled
resource "aws_s3_object" "index" {
  count        = var.enable_website_uploads ? 1 : 0
  bucket       = aws_s3_bucket.website.id
  key          = "index.html"
  source       = "files/index.html"
  content_type = "text/html"
}

resource "aws_s3_object" "about" {
  count        = var.enable_website_uploads ? 1 : 0
  bucket       = aws_s3_bucket.website.id
  key          = "about.html"
  source       = "files/about.html"
  content_type = "text/html"
}

resource "aws_s3_object" "news" {
  count        = var.enable_website_uploads ? 1 : 0
  bucket       = aws_s3_bucket.website.id
  key          = "news.html"
  source       = "files/news.html"
  content_type = "text/html"
}

resource "aws_s3_object" "funds" {
  count        = var.enable_website_uploads ? 1 : 0
  bucket       = aws_s3_bucket.website.id
  key          = "funds.html"
  source       = "files/funds.html"
  content_type = "text/html"
}

resource "aws_s3_object" "style" {
  count        = var.enable_website_uploads ? 1 : 0
  bucket       = aws_s3_bucket.website.id
  key          = "style.css"
  source       = "files/style.css"
  content_type = "text/css"
}
