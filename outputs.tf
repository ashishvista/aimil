output "api_gateway_url" {
  description = "URL of the API Gateway"
  value       = "https://${aws_api_gateway_rest_api.ocr_api.id}.execute-api.${var.aws_region}.amazonaws.com/${aws_api_gateway_stage.prod.stage_name}"
}

output "sagemaker_endpoint_name" {
  description = "Name of the SageMaker endpoint"
  value       = aws_sagemaker_endpoint.ocr_endpoint.name
}

output "upload_bucket_name" {
  description = "Name of the S3 bucket for uploads"
  value       = aws_s3_bucket.upload_bucket.bucket
}

output "sagemaker_bucket_name" {
  description = "Name of the S3 bucket for SageMaker artifacts"
  value       = aws_s3_bucket.sagemaker_bucket.bucket
}

output "pipeline_name" {
  description = "Name of the SageMaker pipeline"
  value       = aws_sagemaker_pipeline.ocr_pipeline.pipeline_name
}

output "lambda_function_names" {
  description = "Names of the Lambda functions"
  value = {
    ocr_processor           = aws_lambda_function.ocr_processor.function_name
    presigned_url_generator = aws_lambda_function.presigned_url_generator.function_name
  }
}

output "lambda_layer_arn" {
  description = "ARN of the Lambda layer"
  value       = aws_lambda_layer_version.dependencies.arn
}

output "api_gateway_log_group" {
  description = "CloudWatch Log Group for API Gateway logs"
  value       = aws_cloudwatch_log_group.api_gateway_logs.name
}

output "api_gateway_stage_name" {
  description = "API Gateway stage name"
  value       = aws_api_gateway_stage.prod.stage_name
}
