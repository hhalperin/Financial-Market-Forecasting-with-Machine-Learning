# AWS Static Website with HTTPS via CloudFront

This repository contains Terraform code to provision:
- A private S3 bucket holding static files.
- An AWS Certificate Manager (ACM) certificate (DNS-validated).
- A CloudFront distribution that serves content over HTTPS.
- A Route53 Alias record (`A`) pointing your domain to CloudFront.

## Prerequisites

1. A registered domain name in Route53 (e.g., `quant-h2.com`) with a **public** hosted zone.
2. AWS CLI / credentials configured on your machine.
3. Terraform 1.0+ installed.
