#!/bin/bash

# API Gateway Logs Viewer
# This script provides easy access to API Gateway logs and monitoring

set -e

echo "📊 API Gateway Logs and Monitoring Dashboard"
echo "=============================================="

# AWS Profile configuration
AWS_PROFILE="test-prod"

# Check AWS credentials
if ! aws sts get-caller-identity --profile $AWS_PROFILE >/dev/null 2>&1; then
    echo "❌ AWS credentials not configured or expired"
    echo "💡 Please run: aws sso login --profile $AWS_PROFILE"
    exit 1
fi

# Get API Gateway details from Terraform
API_NAME=$(terraform output -raw project_name 2>/dev/null || echo "ocr-pipeline")
LOG_GROUP_NAME="/aws/apigateway/${API_NAME}-api"
REGION=$(terraform output -raw aws_region 2>/dev/null || echo "ap-south-1")

echo "🔍 API Gateway Configuration:"
echo "   API Name: $API_NAME"
echo "   Log Group: $LOG_GROUP_NAME" 
echo "   Region: $REGION"
echo "   Profile: $AWS_PROFILE"

# Function to show menu
show_menu() {
    echo ""
    echo "📋 Available Options:"
    echo "====================="
    echo "1️⃣  View Recent Access Logs"
    echo "2️⃣  View Real-time Logs (tail)"
    echo "3️⃣  View Error Logs Only" 
    echo "4️⃣  View CloudWatch Metrics"
    echo "5️⃣  View Method Performance"
    echo "6️⃣  Search Logs by Request ID"
    echo "7️⃣  View Lambda Function Logs"
    echo "8️⃣  Export Logs to File"
    echo "9️⃣  View API Gateway Alarms"
    echo "🔟  Test API Endpoints"
    echo "❌ Exit"
    echo ""
}

# Function to view recent access logs
view_access_logs() {
    echo "📋 Recent API Gateway Access Logs (last 10 minutes):"
    echo "====================================================="
    
    aws logs filter-log-events \
        --log-group-name "$LOG_GROUP_NAME" \
        --start-time $(($(date +%s) * 1000 - 600000)) \
        --profile $AWS_PROFILE \
        --query 'events[].message' \
        --output table || echo "❌ No recent logs found or log group doesn't exist yet"
}

# Function to tail logs in real-time
tail_logs() {
    echo "📡 Tailing API Gateway logs in real-time..."
    echo "==========================================="
    echo "Press Ctrl+C to stop"
    
    aws logs tail "$LOG_GROUP_NAME" \
        --follow \
        --profile $AWS_PROFILE || echo "❌ Could not tail logs"
}

# Function to view error logs only
view_error_logs() {
    echo "❌ Error Logs (last 1 hour):"
    echo "============================"
    
    aws logs filter-log-events \
        --log-group-name "$LOG_GROUP_NAME" \
        --start-time $(($(date +%s) * 1000 - 3600000)) \
        --filter-pattern "{ $.status >= 400 }" \
        --profile $AWS_PROFILE \
        --query 'events[].message' \
        --output table || echo "❌ No error logs found"
}

# Function to view CloudWatch metrics
view_metrics() {
    echo "📊 API Gateway CloudWatch Metrics (last 1 hour):"
    echo "================================================="
    
    # Get 4xx errors
    echo "🔸 4XX Errors:"
    aws cloudwatch get-metric-statistics \
        --namespace AWS/ApiGateway \
        --metric-name 4XXError \
        --dimensions Name=ApiName,Value=${API_NAME}-api Name=Stage,Value=prod \
        --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ) \
        --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
        --period 300 \
        --statistics Sum \
        --profile $AWS_PROFILE \
        --query 'Datapoints[].Sum' \
        --output table
    
    # Get 5xx errors
    echo "🔸 5XX Errors:"
    aws cloudwatch get-metric-statistics \
        --namespace AWS/ApiGateway \
        --metric-name 5XXError \
        --dimensions Name=ApiName,Value=${API_NAME}-api Name=Stage,Value=prod \
        --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ) \
        --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
        --period 300 \
        --statistics Sum \
        --profile $AWS_PROFILE \
        --query 'Datapoints[].Sum' \
        --output table
    
    # Get latency
    echo "🔸 Average Latency (ms):"
    aws cloudwatch get-metric-statistics \
        --namespace AWS/ApiGateway \
        --metric-name Latency \
        --dimensions Name=ApiName,Value=${API_NAME}-api Name=Stage,Value=prod \
        --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ) \
        --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
        --period 300 \
        --statistics Average \
        --profile $AWS_PROFILE \
        --query 'Datapoints[].Average' \
        --output table
}

# Function to search logs by request ID
search_by_request_id() {
    echo "🔍 Search Logs by Request ID"
    echo "============================"
    read -p "Enter Request ID: " request_id
    
    if [ -n "$request_id" ]; then
        aws logs filter-log-events \
            --log-group-name "$LOG_GROUP_NAME" \
            --filter-pattern "{ $.requestId = \"$request_id\" }" \
            --start-time $(($(date +%s) * 1000 - 86400000)) \
            --profile $AWS_PROFILE \
            --query 'events[].message' \
            --output table
    else
        echo "❌ Request ID cannot be empty"
    fi
}

# Function to view Lambda logs
view_lambda_logs() {
    echo "⚡ Lambda Function Logs:"
    echo "======================="
    echo "1. OCR Processor Lambda Logs"
    echo "2. Presigned URL Lambda Logs"
    read -p "Choose option (1-2): " lambda_option
    
    case $lambda_option in
        1)
            aws logs tail "/aws/lambda/${API_NAME}-ocr-processor" \
                --follow \
                --profile $AWS_PROFILE || echo "❌ OCR Processor logs not accessible"
            ;;
        2)
            aws logs tail "/aws/lambda/${API_NAME}-presigned-url" \
                --follow \
                --profile $AWS_PROFILE || echo "❌ Presigned URL logs not accessible"
            ;;
        *)
            echo "❌ Invalid option"
            ;;
    esac
}

# Function to export logs
export_logs() {
    echo "💾 Export Logs to File"
    echo "====================="
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="api_gateway_logs_${TIMESTAMP}.json"
    
    echo "📁 Exporting logs to: $LOG_FILE"
    
    aws logs filter-log-events \
        --log-group-name "$LOG_GROUP_NAME" \
        --start-time $(($(date +%s) * 1000 - 86400000)) \
        --profile $AWS_PROFILE \
        --output json > "$LOG_FILE"
    
    if [ -f "$LOG_FILE" ]; then
        echo "✅ Logs exported successfully to $LOG_FILE"
        echo "📊 File size: $(du -sh "$LOG_FILE" | cut -f1)"
    else
        echo "❌ Failed to export logs"
    fi
}

# Function to view alarms
view_alarms() {
    echo "🚨 API Gateway CloudWatch Alarms:"
    echo "=================================="
    
    aws cloudwatch describe-alarms \
        --alarm-names \
            "${API_NAME}-api-gateway-4xx-errors" \
            "${API_NAME}-api-gateway-5xx-errors" \
            "${API_NAME}-api-gateway-latency" \
        --profile $AWS_PROFILE \
        --query 'MetricAlarms[].[AlarmName,StateValue,StateReason]' \
        --output table || echo "❌ No alarms found"
}

# Function to test API endpoints
test_endpoints() {
    echo "🧪 Test API Endpoints"
    echo "===================="
    
    API_URL=$(terraform output -raw api_gateway_url 2>/dev/null || echo "")
    
    if [ -z "$API_URL" ]; then
        echo "❌ Could not get API Gateway URL from Terraform outputs"
        return 1
    fi
    
    echo "🌐 API Base URL: $API_URL"
    echo ""
    echo "1. Test Upload Endpoint (GET)"
    echo "2. Test Upload Endpoint (POST)"
    echo "3. Test Process Endpoint (POST)"
    echo "4. Test API Health"
    
    read -p "Choose option (1-4): " test_option
    
    case $test_option in
        1)
            echo "📡 Testing GET $API_URL/upload"
            curl -v "$API_URL/upload" || echo "❌ Test failed"
            ;;
        2)
            echo "📡 Testing POST $API_URL/upload"
            curl -X POST "$API_URL/upload" \
                -H "Content-Type: application/json" \
                -d '{"file_extension": "jpg", "content_type": "image/jpeg"}' || echo "❌ Test failed"
            ;;
        3)
            echo "📡 Testing POST $API_URL/process"
            curl -X POST "$API_URL/process" \
                -H "Content-Type: application/json" \
                -d '{"s3_key": "test.jpg", "bucket": "test-bucket"}' || echo "❌ Test failed"
            ;;
        4)
            echo "📡 Testing API Health"
            curl -I "$API_URL" || echo "❌ Health check failed"
            ;;
        *)
            echo "❌ Invalid option"
            ;;
    esac
}

# Main menu loop
while true; do
    show_menu
    read -p "Choose an option: " choice
    
    case $choice in
        1)
            view_access_logs
            ;;
        2)
            tail_logs
            ;;
        3)
            view_error_logs
            ;;
        4)
            view_metrics
            ;;
        5)
            echo "📈 Method Performance - Feature coming soon!"
            ;;
        6)
            search_by_request_id
            ;;
        7)
            view_lambda_logs
            ;;
        8)
            export_logs
            ;;
        9)
            view_alarms
            ;;
        10)
            test_endpoints
            ;;
        "exit"|"quit"|"q"|"x")
            echo "👋 Goodbye!"
            break
            ;;
        *)
            echo "❌ Invalid option. Please try again."
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
done
