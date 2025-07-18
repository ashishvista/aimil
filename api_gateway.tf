# API Gateway REST API
resource "aws_api_gateway_rest_api" "ocr_api" {
  name        = "${var.project_name}-api"
  description = "OCR API for file uploads and processing"

  endpoint_configuration {
    types = ["REGIONAL"]
  }
}

# API Gateway Resource for file upload
resource "aws_api_gateway_resource" "upload_resource" {
  rest_api_id = aws_api_gateway_rest_api.ocr_api.id
  parent_id   = aws_api_gateway_rest_api.ocr_api.root_resource_id
  path_part   = "upload"
}

# API Gateway Resource for OCR processing
resource "aws_api_gateway_resource" "process_resource" {
  rest_api_id = aws_api_gateway_rest_api.ocr_api.id
  parent_id   = aws_api_gateway_rest_api.ocr_api.root_resource_id
  path_part   = "process"
}

# POST method for getting presigned URL
resource "aws_api_gateway_method" "upload_post" {
  rest_api_id   = aws_api_gateway_rest_api.ocr_api.id
  resource_id   = aws_api_gateway_resource.upload_resource.id
  http_method   = "POST"
  authorization = "NONE"
}

# POST method for OCR processing
resource "aws_api_gateway_method" "process_post" {
  rest_api_id   = aws_api_gateway_rest_api.ocr_api.id
  resource_id   = aws_api_gateway_resource.process_resource.id
  http_method   = "POST"
  authorization = "NONE"
}

# Integration for presigned URL generation
resource "aws_api_gateway_integration" "upload_integration" {
  rest_api_id = aws_api_gateway_rest_api.ocr_api.id
  resource_id = aws_api_gateway_resource.upload_resource.id
  http_method = aws_api_gateway_method.upload_post.http_method

  integration_http_method = "POST"
  type                   = "AWS_PROXY"
  uri                    = aws_lambda_function.presigned_url_generator.invoke_arn
}

# Integration for OCR processing
resource "aws_api_gateway_integration" "process_integration" {
  rest_api_id = aws_api_gateway_rest_api.ocr_api.id
  resource_id = aws_api_gateway_resource.process_resource.id
  http_method = aws_api_gateway_method.process_post.http_method

  integration_http_method = "POST"
  type                   = "AWS_PROXY"
  uri                    = aws_lambda_function.ocr_processor.invoke_arn
}

# Lambda permissions for API Gateway
resource "aws_lambda_permission" "api_gateway_upload" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.presigned_url_generator.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.ocr_api.execution_arn}/*/*"
}

resource "aws_lambda_permission" "api_gateway_process" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ocr_processor.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.ocr_api.execution_arn}/*/*"
}

# API Gateway Deployment
resource "aws_api_gateway_deployment" "ocr_api_deployment" {
  depends_on = [
    aws_api_gateway_integration.upload_integration,
    aws_api_gateway_integration.process_integration
  ]

  rest_api_id = aws_api_gateway_rest_api.ocr_api.id
  stage_name  = "prod"
}

# CORS configuration
resource "aws_api_gateway_method" "upload_options" {
  rest_api_id   = aws_api_gateway_rest_api.ocr_api.id
  resource_id   = aws_api_gateway_resource.upload_resource.id
  http_method   = "OPTIONS"
  authorization = "NONE"
}

resource "aws_api_gateway_method" "process_options" {
  rest_api_id   = aws_api_gateway_rest_api.ocr_api.id
  resource_id   = aws_api_gateway_resource.process_resource.id
  http_method   = "OPTIONS"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "upload_options_integration" {
  rest_api_id = aws_api_gateway_rest_api.ocr_api.id
  resource_id = aws_api_gateway_resource.upload_resource.id
  http_method = aws_api_gateway_method.upload_options.http_method

  type = "MOCK"
  request_templates = {
    "application/json" = jsonencode({
      statusCode = 200
    })
  }
}

resource "aws_api_gateway_integration" "process_options_integration" {
  rest_api_id = aws_api_gateway_rest_api.ocr_api.id
  resource_id = aws_api_gateway_resource.process_resource.id
  http_method = aws_api_gateway_method.process_options.http_method

  type = "MOCK"
  request_templates = {
    "application/json" = jsonencode({
      statusCode = 200
    })
  }
}

# Method responses for CORS
resource "aws_api_gateway_method_response" "upload_options_response" {
  rest_api_id = aws_api_gateway_rest_api.ocr_api.id
  resource_id = aws_api_gateway_resource.upload_resource.id
  http_method = aws_api_gateway_method.upload_options.http_method
  status_code = "200"

  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = true
    "method.response.header.Access-Control-Allow-Methods" = true
    "method.response.header.Access-Control-Allow-Origin"  = true
  }
}

resource "aws_api_gateway_method_response" "process_options_response" {
  rest_api_id = aws_api_gateway_rest_api.ocr_api.id
  resource_id = aws_api_gateway_resource.process_resource.id
  http_method = aws_api_gateway_method.process_options.http_method
  status_code = "200"

  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = true
    "method.response.header.Access-Control-Allow-Methods" = true
    "method.response.header.Access-Control-Allow-Origin"  = true
  }
}

# Integration responses for CORS
resource "aws_api_gateway_integration_response" "upload_options_integration_response" {
  rest_api_id = aws_api_gateway_rest_api.ocr_api.id
  resource_id = aws_api_gateway_resource.upload_resource.id
  http_method = aws_api_gateway_method.upload_options.http_method
  status_code = aws_api_gateway_method_response.upload_options_response.status_code

  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'"
    "method.response.header.Access-Control-Allow-Methods" = "'GET,OPTIONS,POST,PUT'"
    "method.response.header.Access-Control-Allow-Origin"  = "'*'"
  }
}

resource "aws_api_gateway_integration_response" "process_options_integration_response" {
  rest_api_id = aws_api_gateway_rest_api.ocr_api.id
  resource_id = aws_api_gateway_resource.process_resource.id
  http_method = aws_api_gateway_method.process_options.http_method
  status_code = aws_api_gateway_method_response.process_options_response.status_code

  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'"
    "method.response.header.Access-Control-Allow-Methods" = "'GET,OPTIONS,POST,PUT'"
    "method.response.header.Access-Control-Allow-Origin"  = "'*'"
  }
}
