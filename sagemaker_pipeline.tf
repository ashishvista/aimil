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
    aws_s3_object.requirements
  ]
}
