########################################################################
# cloudfront.tf
########################################################################

resource "aws_cloudfront_origin_access_control" "oac" {
  name                              = "${var.domain_name}-oac"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
  origin_access_control_origin_type = "s3"
}

resource "aws_cloudfront_distribution" "cdn" {
  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "index.html"

  aliases = [
    var.domain_name,
    "www.${var.domain_name}"
  ]

  origin {
    domain_name              = aws_s3_bucket.website.bucket_regional_domain_name
    origin_id                = "s3-${var.domain_name}"
    origin_access_control_id = aws_cloudfront_origin_access_control.oac.id
  }

  default_cache_behavior {
    target_origin_id       = "s3-${var.domain_name}"
    viewer_protocol_policy = "redirect-to-https"

    allowed_methods = ["GET", "HEAD"]
    cached_methods  = ["GET", "HEAD"]

    forwarded_values {
      query_string = false

      cookies {
        forward = "none"
      }
    }

    min_ttl     = 0
    default_ttl = 3600
    max_ttl     = 86400
  }

  # If you don't need to distribute content globally, limit the price class to reduce cost
  price_class = "PriceClass_100"

  # Link the validated certificate
  viewer_certificate {
    acm_certificate_arn           = aws_acm_certificate_validation.cert_validation.certificate_arn
    ssl_support_method             = "sni-only"
    minimum_protocol_version       = "TLSv1.2_2021"
    cloudfront_default_certificate = false
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  tags = {
    Environment = var.environment
    Name        = "${var.domain_name}-CF"
  }
}

# S3 Bucket Policy to allow CloudFront OAC to fetch objects
resource "aws_s3_bucket_policy" "website_policy" {
  # Optionally ensure the distribution is created before the bucket policy is applied
  depends_on = [aws_cloudfront_distribution.cdn]

  bucket = aws_s3_bucket.website.id

  policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        # For OAC, the recommended principal is the CloudFront service,
        # not the old OAI user ARN.
        Principal = {
          Service = "cloudfront.amazonaws.com"
        }
        Action   = [ "s3:GetObject" ]
        Resource = [ "${aws_s3_bucket.website.arn}/*" ]

        Condition = {
          StringEquals = {
            # Make sure the SourceArn matches the distribution's ARN
            "AWS:SourceArn" = aws_cloudfront_distribution.cdn.arn
          }
        }
      }
    ]
  })
}
