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

  depends_on = [aws_s3_object.model_artifact, aws_s3_object.inference_script]
}

# Null resource to verify model creation
resource "null_resource" "verify_model_creation" {
  triggers = {
    model_name = aws_sagemaker_model.ocr_model.name
  }

  provisioner "local-exec" {
    command = <<-EOT
      echo "🔍 Verifying SageMaker model creation..."
      echo "Model Name: ${aws_sagemaker_model.ocr_model.name}"
      echo "Model ARN: ${aws_sagemaker_model.ocr_model.arn}"
      echo "Model Data URL: s3://${aws_s3_bucket.sagemaker_bucket.bucket}/models/model.tar.gz"
      
      # Verify the model exists
      aws sagemaker describe-model --model-name ${aws_sagemaker_model.ocr_model.name} --region ${var.aws_region} --profile test-prod > /dev/null
      
      if [ $? -eq 0 ]; then
        echo "✅ SageMaker model verified successfully"
      else
        echo "❌ SageMaker model verification failed"
        exit 1
      fi
    EOT
  }

  depends_on = [aws_sagemaker_model.ocr_model]
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

# Null resource to verify endpoint deployment and readiness
resource "null_resource" "verify_endpoint_deployment" {
  triggers = {
    endpoint_name = aws_sagemaker_endpoint.ocr_endpoint.name
    config_name   = aws_sagemaker_endpoint_configuration.ocr_endpoint_config.name
  }

  provisioner "local-exec" {
    command = <<-EOT
      echo "🚀 Verifying SageMaker endpoint deployment..."
      echo "Endpoint Name: ${aws_sagemaker_endpoint.ocr_endpoint.name}"
      echo "Config Name: ${aws_sagemaker_endpoint_configuration.ocr_endpoint_config.name}"
      
      # Wait for endpoint to be in service
      echo "⏳ Waiting for endpoint to be ready (this may take several minutes)..."
      aws sagemaker wait endpoint-in-service \
        --endpoint-name ${aws_sagemaker_endpoint.ocr_endpoint.name} \
        --region ${var.aws_region} \
        --profile test-prod
      
      if [ $? -eq 0 ]; then
        echo "✅ SageMaker endpoint is ready and in service"
        
        # Get endpoint status
        ENDPOINT_STATUS=$(aws sagemaker describe-endpoint \
          --endpoint-name ${aws_sagemaker_endpoint.ocr_endpoint.name} \
          --region ${var.aws_region} \
          --profile test-prod \
          --query 'EndpointStatus' \
          --output text)
        
        echo "📊 Endpoint Status: $ENDPOINT_STATUS"
      else
        echo "❌ SageMaker endpoint deployment failed or timed out"
        exit 1
      fi
    EOT
  }

  depends_on = [aws_sagemaker_endpoint.ocr_endpoint]
}
