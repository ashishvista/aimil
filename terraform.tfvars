# AWS region
aws_region = "us-east-1"

# Project configuration
project_name = "ocr-pipeline"
model_name = "ocr-model"

# Instance types
endpoint_instance_type = "ml.m5.large"
training_instance_type = "ml.m5.xlarge"

# Docker images (update these based on your region)
ecr_image_uri = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.0-gpu-py38"
inference_image_uri = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.12.0-gpu-py38"
