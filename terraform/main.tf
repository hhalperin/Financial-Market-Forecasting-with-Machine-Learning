data "aws_iam_policy_document" "lambda_assume_role_policy" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "batch_assume_role_policy" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["batch.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "sfn_assume_role_policy" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }
  }
}

# Zip the lambda code from ../src directory
resource "null_resource" "zip_lambda_code" {
  # Change the command if you have different code structure
  provisioner "local-exec" {
    command = "cd .. && zip -r infrastructure/lambda_package.zip src"
  }

  triggers = {
    # Re-run on code changes by including a checksum if desired
    last_modified = timestamp()
  }
}

# Upload the zipped code to S3 for lambda (optional)
resource "aws_s3_object" "lambda_code_object" {
  bucket = aws_s3_bucket.lambda_code_bucket.bucket
  key    = "my_lambda_code.zip"
  source = "${path.module}/lambda_package.zip"

  depends_on = [null_resource.zip_lambda_code]
}
