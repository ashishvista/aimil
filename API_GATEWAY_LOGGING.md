# API Gateway Comprehensive Logging Configuration

## 🎯 Overview

This configuration enables **comprehensive logging and monitoring** for the OCR Pipeline API Gateway, including:
- **Access logs** with detailed request/response information
- **CloudWatch metrics** for performance monitoring  
- **Error tracking** and alerting
- **X-Ray tracing** for distributed tracing
- **Method-level logging** with different log levels

## 🏗️ Infrastructure Components

### 1. IAM Role for API Gateway Logging
```hcl
resource "aws_iam_role" "api_gateway_logs_role" {
  # Allows API Gateway to write logs to CloudWatch
}
```

### 2. CloudWatch Log Group
```hcl
resource "aws_cloudwatch_log_group" "api_gateway_logs" {
  name              = "/aws/apigateway/${var.project_name}-api"
  retention_in_days = 14  # Customize retention period
}
```

### 3. API Gateway Stage with Access Logs
```hcl
resource "aws_api_gateway_stage" "prod" {
  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway_logs.arn
    format = jsonencode({
      # Comprehensive log format with all relevant fields
    })
  }
  
  xray_tracing_enabled = true  # Enable X-Ray tracing
}
```

### 4. Method Settings for Detailed Logging
```hcl
resource "aws_api_gateway_method_settings" "all" {
  settings {
    metrics_enabled    = true
    logging_level      = "INFO"    # INFO level captures all requests/responses
    data_trace_enabled = true      # Includes request/response bodies
    throttling_rate_limit  = 1000  # Rate limiting
    throttling_burst_limit = 2000
  }
}
```

### 5. CloudWatch Alarms
```hcl
# Monitors 4xx errors, 5xx errors, and high latency
# Triggers alerts when thresholds are exceeded
```

## 📊 Log Format Details

The access logs capture comprehensive information in JSON format:

### Request Information
- `requestId` - Unique identifier for each request
- `requestTime` - Timestamp of the request
- `httpMethod` - GET, POST, PUT, etc.
- `resourcePath` - API endpoint path
- `sourceIp` - Client IP address
- `userAgent` - Client user agent string

### Response Information  
- `status` - HTTP status code (200, 400, 500, etc.)
- `responseLength` - Size of response body
- `responseTime` - Total request processing time
- `protocol` - HTTP protocol version

### Error Information
- `error.message` - Error message if any
- `error.responseType` - Type of error response
- `integrationError.message` - Backend integration errors
- `integrationError.status` - Integration status

### Performance Metrics
- `responseTime` - End-to-end latency
- `integration.latency` - Backend integration latency
- `waf.latency` - WAF processing time (if applicable)

## 🔍 Logging Levels

### INFO Level (Enabled)
- ✅ **Request/Response Headers**
- ✅ **Request/Response Bodies** (when data_trace_enabled = true)
- ✅ **Execution Logs**
- ✅ **Error Messages**
- ✅ **Performance Metrics**

### ERROR Level (Included in INFO)
- ✅ **All ERROR level logs**
- ✅ **Exception Stack Traces**
- ✅ **Integration Failures**
- ✅ **Authentication Errors**

## 🛠️ Monitoring Tools

### 1. Interactive Log Viewer Script
```bash
./view_api_logs.sh
```

**Features:**
- Real-time log tailing
- Error log filtering
- CloudWatch metrics viewing
- Request ID search
- Lambda function logs
- Log export functionality
- API endpoint testing

### 2. CloudWatch Dashboard
Access via AWS Console:
- **Log Group**: `/aws/apigateway/ocr-pipeline-api`
- **Metrics**: AWS/ApiGateway namespace
- **Alarms**: 4xx/5xx errors and latency monitoring

### 3. X-Ray Tracing
- **Service Map**: Visual representation of request flow
- **Trace Analysis**: Detailed request tracing through services
- **Performance Insights**: Bottleneck identification

## 📈 Key Metrics Monitored

### Performance Metrics
- **Latency**: Average response time per endpoint
- **Count**: Number of requests per time period
- **Integration Latency**: Backend processing time

### Error Metrics
- **4XX Errors**: Client errors (bad requests, unauthorized, etc.)
- **5XX Errors**: Server errors (internal server error, bad gateway, etc.)
- **Throttling**: Rate limit exceeded events

### Business Metrics
- **Upload Success Rate**: Successful file uploads
- **OCR Processing Success Rate**: Successful text extractions
- **API Availability**: Overall API uptime

## 🚨 Alerting Configuration

### Current Alarms
1. **4XX Errors > 10** in 10 minutes
2. **5XX Errors > 5** in 10 minutes  
3. **Average Latency > 5 seconds**

### Custom Alerting (Optional)
To add SNS notifications, update alarm resources:
```hcl
resource "aws_cloudwatch_metric_alarm" "api_gateway_4xx_errors" {
  alarm_actions = [aws_sns_topic.alerts.arn]  # Add SNS topic
}
```

## 🔧 Log Analysis Commands

### View Recent Logs
```bash
aws logs filter-log-events \
  --log-group-name "/aws/apigateway/ocr-pipeline-api" \
  --start-time $(date -d '1 hour ago' +%s)000 \
  --profile test-prod
```

### Filter Error Logs
```bash
aws logs filter-log-events \
  --log-group-name "/aws/apigateway/ocr-pipeline-api" \
  --filter-pattern "{ $.status >= 400 }" \
  --profile test-prod
```

### Search by Request ID
```bash
aws logs filter-log-events \
  --log-group-name "/aws/apigateway/ocr-pipeline-api" \
  --filter-pattern "{ $.requestId = \"your-request-id\" }" \
  --profile test-prod
```

### Export Logs
```bash
aws logs create-export-task \
  --log-group-name "/aws/apigateway/ocr-pipeline-api" \
  --from $(date -d '1 day ago' +%s)000 \
  --to $(date +%s)000 \
  --destination "s3://your-log-bucket/api-gateway-logs/" \
  --profile test-prod
```

## 📋 Troubleshooting

### Common Issues

1. **Logs Not Appearing**
   - Check IAM role permissions
   - Verify CloudWatch log group exists
   - Ensure API Gateway account settings are configured

2. **Missing Request/Response Bodies**
   - Verify `data_trace_enabled = true` in method settings
   - Check logging level is set to INFO

3. **High Log Volume/Costs**
   - Adjust log retention period (currently 14 days)
   - Consider sampling for high-traffic APIs
   - Use log filtering to reduce noise

### Debug Commands
```bash
# Check API Gateway account settings
aws apigateway get-account --profile test-prod

# Verify log group exists
aws logs describe-log-groups \
  --log-group-name-prefix "/aws/apigateway" \
  --profile test-prod

# Test logging configuration
curl -v "$(terraform output -raw api_gateway_url)/upload"
```

## 🎯 Benefits

### ✅ **Comprehensive Visibility**
- Full request/response lifecycle tracking
- Detailed error information and stack traces
- Performance metrics and bottleneck identification

### ✅ **Proactive Monitoring**
- Real-time error alerting
- Performance threshold monitoring  
- Automated anomaly detection

### ✅ **Debugging Capabilities**
- Request ID correlation across services
- X-Ray distributed tracing
- Lambda function log correlation

### ✅ **Compliance & Audit**
- Complete audit trail of API usage
- Request/response logging for compliance
- Retention policies for data governance

## 🚀 Next Steps

1. **Deploy Configuration**
   ```bash
   terraform apply -auto-approve
   ```

2. **Test Logging**
   ```bash
   ./view_api_logs.sh
   ```

3. **Set Up Notifications** (Optional)
   - Create SNS topic for alerts
   - Configure email/Slack notifications
   - Update alarm actions

4. **Create Custom Dashboards** (Optional)
   - CloudWatch dashboard for business metrics
   - Grafana integration for advanced visualization
   - Custom log analysis with ELK stack

This comprehensive logging configuration provides complete visibility into your OCR Pipeline API Gateway, enabling proactive monitoring, debugging, and performance optimization! 🎉
