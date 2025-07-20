# Upload training script and model code to S3
resource "aws_s3_object" "training_script" {
  bucket = aws_s3_bucket.sagemaker_bucket.bucket
  key    = "code/train.py"
  source = "${path.module}/scripts/train.py"
  etag   = filemd5("${path.module}/scripts/train.py")

  depends_on = [aws_s3_bucket.sagemaker_bucket]
}

resource "aws_s3_object" "inference_script" {
  bucket = aws_s3_bucket.sagemaker_bucket.bucket
  key    = "code/inference.py"
  source = "${path.module}/scripts/inference.py"
  etag   = filemd5("${path.module}/scripts/inference.py")

  depends_on = [aws_s3_bucket.sagemaker_bucket]
}

resource "aws_s3_object" "requirements" {
  bucket = aws_s3_bucket.sagemaker_bucket.bucket
  key    = "code/requirements.txt"
  source = "${path.module}/scripts/requirements.txt"
  etag   = filemd5("${path.module}/scripts/requirements.txt")

  depends_on = [aws_s3_bucket.sagemaker_bucket]
}

# Create model artifact if it doesn't exist
resource "null_resource" "create_model_artifact" {
  triggers = {
    always_run = timestamp()
  }

  provisioner "local-exec" {
    command = <<-EOT
      if [ ! -f "${path.module}/model.tar.gz" ]; then
        echo "Creating placeholder model.tar.gz..."
        mkdir -p temp_model
        cat > temp_model/model.py << 'EOF'
# Placeholder OCR model for SageMaker endpoint
import json
import logging

logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """Load the model for inference"""
    logger.info("Loading OCR model...")
    return {"status": "model_loaded", "model_dir": model_dir}

def input_fn(request_body, content_type):
    """Parse input data for inference"""
    if content_type == 'application/json':
        return json.loads(request_body)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Run inference"""
    logger.info("Running OCR inference...")
    return {
        "text": "Sample extracted text from OCR model",
        "confidence": 0.95,
        "status": "success"
    }

def output_fn(prediction, accept):
    """Format the output"""
    if accept == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
EOF
        tar -czf model.tar.gz -C temp_model .
        rm -rf temp_model
        echo "✅ Created model.tar.gz ($(du -sh model.tar.gz | cut -f1))"
      else
        echo "✅ model.tar.gz already exists ($(du -sh model.tar.gz | cut -f1))"
      fi
    EOT
  }
}

# Upload model artifact to S3
resource "aws_s3_object" "model_artifact" {
  bucket = aws_s3_bucket.sagemaker_bucket.bucket
  key    = "models/model.tar.gz"
  source = "${path.module}/model.tar.gz"
  etag   = filemd5("${path.module}/model.tar.gz")

  depends_on = [aws_s3_bucket.sagemaker_bucket, null_resource.create_model_artifact]
}

# SageMaker Pipeline Definition
resource "aws_sagemaker_pipeline" "ocr_pipeline" {
  pipeline_name        = "${var.project_name}-pipeline"
  pipeline_display_name = "OCR-Model-Training-Pipeline"
  role_arn             = aws_iam_role.sagemaker_pipeline_role.arn

  pipeline_definition = jsonencode({
    Version = "2020-12-01"
    Metadata = {
      DefaultPipeline = "true"
    }
    Parameters = [
      {
        Name         = "ProcessingInstanceType"
        Type         = "String"
        DefaultValue = var.training_instance_type
      },
      {
        Name         = "TrainingInstanceType"
        Type         = "String"
        DefaultValue = var.training_instance_type
      },
      {
        Name         = "ModelApprovalStatus"
        Type         = "String"
        DefaultValue = "PendingManualApproval"
      },
      {
        Name         = "InputDataUrl"
        Type         = "String"
        DefaultValue = "s3://${aws_s3_bucket.sagemaker_bucket.bucket}/data/"
      }
    ]
    Steps = [
      {
        Name = "TrainModel"
        Type = "Training"
        Arguments = {
          AlgorithmSpecification = {
            TrainingImage = var.ecr_image_uri
            TrainingInputMode = "File"
          }
          InputDataConfig = [
            {
              ChannelName = "training"
              DataSource = {
                S3DataSource = {
                  S3DataType = "S3Prefix"
                  S3Uri = "s3://${aws_s3_bucket.sagemaker_bucket.bucket}/data/"
                  S3DataDistributionType = "FullyReplicated"
                }
              }
              ContentType = "application/json"
              CompressionType = "None"
            }
          ]
          OutputDataConfig = {
            S3OutputPath = "s3://${aws_s3_bucket.sagemaker_bucket.bucket}/models/"
          }
          ResourceConfig = {
            InstanceCount = 1
            InstanceType = var.training_instance_type
            VolumeSizeInGB = 30
          }
          RoleArn = aws_iam_role.sagemaker_execution_role.arn
          StoppingCondition = {
            MaxRuntimeInSeconds = 86400
          }
          HyperParameters = {
            epochs = "10"
            batch_size = "32"
            learning_rate = "0.001"
          }
        }
      }
    ]
  })

  depends_on = [
    aws_s3_object.training_script,
    aws_s3_object.inference_script,
    aws_s3_object.requirements,
    aws_s3_object.model_artifact
  ]
}
