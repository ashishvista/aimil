variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-south-1"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "ocr-pipeline"
}

variable "model_name" {
  description = "Name of the OCR model"
  type        = string
  default     = "ocr-model"
}

variable "endpoint_instance_type" {
  description = "Instance type for SageMaker endpoint"
  type        = string
  default     = "ml.m5.large"
}

variable "training_instance_type" {
  description = "Instance type for SageMaker training"
  type        = string
  default     = "ml.m5.xlarge"
}

variable "ecr_image_uri" {
  description = "ECR image URI for the OCR model"
  type        = string
  default     = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.0-gpu-py38"
}

variable "inference_image_uri" {
  description = "ECR image URI for inference"
  type        = string
  default     = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.12.0-gpu-py38"
}

variable "custom_inference_image" {
  description = "Custom Docker image URI for SageMaker inference with Tesseract"
  type        = string
  default     = "913197190703.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-ocr-tesseract:latest"
}
