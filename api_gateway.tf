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

  triggers = {
    redeployment = sha1(jsonencode([
      aws_api_gateway_resource.upload_resource.id,
      aws_api_gateway_resource.process_resource.id,
      aws_api_gateway_method.upload_post.id,
      aws_api_gateway_method.process_post.id,
      aws_api_gateway_integration.upload_integration.id,
      aws_api_gateway_integration.process_integration.id,
    ]))
  }

  lifecycle {
    create_before_destroy = true
  }
}

# API Gateway Stage with comprehensive logging
resource "aws_api_gateway_stage" "prod" {
  deployment_id = aws_api_gateway_deployment.ocr_api_deployment.id
  rest_api_id   = aws_api_gateway_rest_api.ocr_api.id
  stage_name    = "prod"

  # Enable comprehensive logging
  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway_logs.arn
    format = jsonencode({
      requestId        = "$context.requestId"
      requestTime      = "$context.requestTime"
      requestTimeEpoch = "$context.requestTimeEpoch"
      httpMethod       = "$context.httpMethod"
      resourcePath     = "$context.resourcePath"
      status          = "$context.status"
      protocol        = "$context.protocol"
      responseLength  = "$context.responseLength"
      requestLength   = "$context.requestLength"
      responseTime    = "$context.responseTime"
      sourceIp        = "$context.identity.sourceIp"
      userAgent       = "$context.identity.userAgent"
      error          = {
        message      = "$context.error.message"
        messageString = "$context.error.messageString"
        responseType = "$context.error.responseType"
      }
      integrationError = {
        message        = "$context.integration.error"
        status         = "$context.integration.status"
        latency        = "$context.integration.latency"
        requestId      = "$context.integration.requestId"
        integrationStatus = "$context.integration.integrationStatus"
      }
      waf = {
        status     = "$context.waf.status"
        latency    = "$context.waf.latency"
        response   = "$context.waf.response"
      }
    })
  }

  # Enable method-level logging settings
  xray_tracing_enabled = true

  depends_on = [
    aws_api_gateway_account.main,
    aws_cloudwatch_log_group.api_gateway_logs
  ]

  tags = {
    Name = "${var.project_name}-api-stage-prod"
  }
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

# IAM Role for API Gateway logging
resource "aws_iam_role" "api_gateway_logs_role" {
  name = "${var.project_name}-api-gateway-logs-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "apigateway.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "api_gateway_logs_policy" {
  role       = aws_iam_role.api_gateway_logs_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonAPIGatewayPushToCloudWatchLogs"
}

# API Gateway Account settings for logging
resource "aws_api_gateway_account" "main" {
  cloudwatch_role_arn = aws_iam_role.api_gateway_logs_role.arn
}

# CloudWatch Log Group for API Gateway
resource "aws_cloudwatch_log_group" "api_gateway_logs" {
  name              = "/aws/apigateway/${var.project_name}-api"
  retention_in_days = 14

  tags = {
    Name = "${var.project_name}-api-gateway-logs"
  }
}

# API Gateway Method Settings for detailed logging
resource "aws_api_gateway_method_settings" "all" {
  rest_api_id = aws_api_gateway_rest_api.ocr_api.id
  stage_name  = aws_api_gateway_stage.prod.stage_name
  method_path = "*/*"

  settings {
    # Enable detailed CloudWatch metrics
    metrics_enabled = true
    
    # Enable detailed logging
    logging_level   = "INFO"
    data_trace_enabled = true
    
    # Throttling settings
    throttling_rate_limit  = 1000
    throttling_burst_limit = 2000
    
    # Caching settings (disabled for development)
    caching_enabled = false
    
    # Request/response logging
    require_authorization_for_cache_control = false
  }

  depends_on = [aws_api_gateway_stage.prod]
}

# Additional method settings for upload endpoint
resource "aws_api_gateway_method_settings" "upload_detailed" {
  rest_api_id = aws_api_gateway_rest_api.ocr_api.id
  stage_name  = aws_api_gateway_stage.prod.stage_name
  method_path = "${aws_api_gateway_resource.upload_resource.path_part}/POST"

  settings {
    metrics_enabled    = true
    logging_level      = "INFO"
    data_trace_enabled = true
    
    # More detailed settings for upload endpoint
    throttling_rate_limit  = 500
    throttling_burst_limit = 1000
  }

  depends_on = [aws_api_gateway_stage.prod]
}

# Additional method settings for process endpoint  
resource "aws_api_gateway_method_settings" "process_detailed" {
  rest_api_id = aws_api_gateway_rest_api.ocr_api.id
  stage_name  = aws_api_gateway_stage.prod.stage_name
  method_path = "${aws_api_gateway_resource.process_resource.path_part}/POST"

  settings {
    metrics_enabled    = true
    logging_level      = "INFO" 
    data_trace_enabled = true
    
    # OCR processing may take longer
    throttling_rate_limit  = 100
    throttling_burst_limit = 200
  }

  depends_on = [aws_api_gateway_stage.prod]
}

# CloudWatch Alarms for API Gateway monitoring
resource "aws_cloudwatch_metric_alarm" "api_gateway_4xx_errors" {
  alarm_name          = "${var.project_name}-api-gateway-4xx-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "4XXError"
  namespace           = "AWS/ApiGateway"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "This metric monitors 4xx errors in API Gateway"
  alarm_actions       = [] # Add SNS topic ARN here if you want notifications

  dimensions = {
    ApiName   = aws_api_gateway_rest_api.ocr_api.name
    Stage     = aws_api_gateway_stage.prod.stage_name
  }

  tags = {
    Name = "${var.project_name}-api-4xx-errors-alarm"
  }
}

resource "aws_cloudwatch_metric_alarm" "api_gateway_5xx_errors" {
  alarm_name          = "${var.project_name}-api-gateway-5xx-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "5XXError"
  namespace           = "AWS/ApiGateway"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  alarm_description   = "This metric monitors 5xx errors in API Gateway"
  alarm_actions       = [] # Add SNS topic ARN here if you want notifications

  dimensions = {
    ApiName   = aws_api_gateway_rest_api.ocr_api.name
    Stage     = aws_api_gateway_stage.prod.stage_name
  }

  tags = {
    Name = "${var.project_name}-api-5xx-errors-alarm"
  }
}

resource "aws_cloudwatch_metric_alarm" "api_gateway_latency" {
  alarm_name          = "${var.project_name}-api-gateway-latency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Latency"
  namespace           = "AWS/ApiGateway"
  period              = 300
  statistic           = "Average"
  threshold           = 5000 # 5 seconds
  alarm_description   = "This metric monitors high latency in API Gateway"
  alarm_actions       = [] # Add SNS topic ARN here if you want notifications

  dimensions = {
    ApiName   = aws_api_gateway_rest_api.ocr_api.name
    Stage     = aws_api_gateway_stage.prod.stage_name
  }

  tags = {
    Name = "${var.project_name}-api-latency-alarm"
  }
}
