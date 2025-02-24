########################################################################
# outputs.tf
########################################################################
output "s3_bucket_name" {
  description = "The name of the S3 bucket."
  value       = aws_s3_bucket.website.id
}

output "cloudfront_domain_name" {
  description = "CloudFront domain name (useful for debugging)."
  value       = aws_cloudfront_distribution.cdn.domain_name
}

output "website_domain" {
  description = "Your public domain name"
  value       = var.domain_name
}
