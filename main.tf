terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region  = var.aws_region
  profile = "test-prod"
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# S3 bucket for SageMaker artifacts
resource "aws_s3_bucket" "sagemaker_bucket" {
  bucket = "${var.project_name}-sagemaker-${random_string.suffix.result}"
}

resource "aws_s3_bucket_versioning" "sagemaker_bucket_versioning" {
  bucket = aws_s3_bucket.sagemaker_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "sagemaker_bucket_encryption" {
  bucket = aws_s3_bucket.sagemaker_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# S3 bucket for file uploads
resource "aws_s3_bucket" "upload_bucket" {
  bucket = "${var.project_name}-uploads-${random_string.suffix.result}"
}

resource "aws_s3_bucket_versioning" "upload_bucket_versioning" {
  bucket = aws_s3_bucket.upload_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "upload_bucket_encryption" {
  bucket = aws_s3_bucket.upload_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_cors_configuration" "upload_bucket_cors" {
  bucket = aws_s3_bucket.upload_bucket.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "POST", "PUT"]
    allowed_origins = ["*"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
}

resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}
