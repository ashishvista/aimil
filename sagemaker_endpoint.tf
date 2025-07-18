# SageMaker Model
resource "aws_sagemaker_model" "ocr_model" {
  name               = "${var.project_name}-model-${random_string.suffix.result}"
  execution_role_arn = aws_iam_role.sagemaker_execution_role.arn

  primary_container {
    image          = var.inference_image_uri
    model_data_url = "s3://${aws_s3_bucket.sagemaker_bucket.bucket}/models/model.tar.gz"
    environment = {
      SAGEMAKER_PROGRAM = "inference.py"
      SAGEMAKER_SUBMIT_DIRECTORY = "s3://${aws_s3_bucket.sagemaker_bucket.bucket}/code/"
    }
  }

  depends_on = [aws_sagemaker_pipeline.ocr_pipeline]
}

# SageMaker Endpoint Configuration
resource "aws_sagemaker_endpoint_configuration" "ocr_endpoint_config" {
  name = "${var.project_name}-endpoint-config-${random_string.suffix.result}"

  production_variants {
    variant_name           = "primary"
    model_name            = aws_sagemaker_model.ocr_model.name
    initial_instance_count = 1
    instance_type         = var.endpoint_instance_type
    initial_variant_weight = 1
  }

  depends_on = [aws_sagemaker_model.ocr_model]
}

# SageMaker Endpoint
resource "aws_sagemaker_endpoint" "ocr_endpoint" {
  name                 = "${var.project_name}-endpoint-${random_string.suffix.result}"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.ocr_endpoint_config.name

  depends_on = [aws_sagemaker_endpoint_configuration.ocr_endpoint_config]
}
