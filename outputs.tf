output "api_gateway_url" {
  description = "URL of the API Gateway"
  value       = aws_api_gateway_deployment.ocr_api_deployment.invoke_url
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
    ocr_processor         = aws_lambda_function.ocr_processor.function_name
    presigned_url_generator = aws_lambda_function.presigned_url_generator.function_name
  }
}
