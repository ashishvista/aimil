# AWS region
aws_region = "ap-south-1"

# Project configuration
project_name = "ocr-pipeline"
model_name = "ocr-model"

# Instance types
endpoint_instance_type = "ml.m5.large"
training_instance_type = "ml.m5.xlarge"

# Docker images (update these based on your region)
ecr_image_uri = "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:1.12.0-gpu-py38"
inference_image_uri = "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-inference:1.12.0-gpu-py38"
