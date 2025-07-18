# Lambda function for OCR processing
resource "aws_lambda_function" "ocr_processor" {
  filename         = data.archive_file.lambda_zip.output_path
  function_name    = "${var.project_name}-ocr-processor"
  role            = aws_iam_role.lambda_execution_role.arn
  handler         = "lambda_function.lambda_handler"
  runtime         = "python3.9"
  timeout         = 300
  memory_size     = 512

  environment {
    variables = {
      SAGEMAKER_ENDPOINT_NAME = aws_sagemaker_endpoint.ocr_endpoint.name
      UPLOAD_BUCKET          = aws_s3_bucket.upload_bucket.bucket
    }
  }

  depends_on = [aws_sagemaker_endpoint.ocr_endpoint]
}

# Lambda function for generating presigned URLs
resource "aws_lambda_function" "presigned_url_generator" {
  filename         = data.archive_file.presigned_lambda_zip.output_path
  function_name    = "${var.project_name}-presigned-url"
  role            = aws_iam_role.lambda_execution_role.arn
  handler         = "presigned_url.lambda_handler"
  runtime         = "python3.9"
  timeout         = 30

  environment {
    variables = {
      UPLOAD_BUCKET = aws_s3_bucket.upload_bucket.bucket
    }
  }
}

# Archive Lambda functions
data "archive_file" "lambda_zip" {
  type        = "zip"
  source_file = "${path.module}/lambda/lambda_function.py"
  output_path = "${path.module}/lambda_function.zip"
}

data "archive_file" "presigned_lambda_zip" {
  type        = "zip"
  source_file = "${path.module}/lambda/presigned_url.py"
  output_path = "${path.module}/presigned_url.zip"
}
