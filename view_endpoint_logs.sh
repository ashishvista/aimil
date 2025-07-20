#!/bin/bash
# view_endpoint_logs.sh - Check SageMaker endpoint inference logs

set -e

AWS_PROFILE="test-prod"
ENDPOINT_NAME=""

# Try to get endpoint name from Terraform outputs
if [ -f terraform.tfstate ]; then
    ENDPOINT_NAME=$(terraform output -raw sagemaker_endpoint_name 2>/dev/null || echo "")
fi

# If we couldn't get it from Terraform, try to find it
if [ -z "$ENDPOINT_NAME" ]; then
    echo "🔍 Trying to find SageMaker endpoint..."
    ENDPOINTS=$(aws sagemaker list-endpoints --profile $AWS_PROFILE --query 'Endpoints[?contains(EndpointName, `ocr`) || contains(EndpointName, `pipeline`)].EndpointName' --output text 2>/dev/null || echo "")
    
    if [ -n "$ENDPOINTS" ]; then
        # If multiple endpoints found, use the first one
        ENDPOINT_NAME=$(echo $ENDPOINTS | awk '{print $1}')
        echo "✅ Found endpoint: $ENDPOINT_NAME"
    else
        echo "❌ Could not find SageMaker endpoint automatically"
        echo "📋 Available endpoints:"
        aws sagemaker list-endpoints --profile $AWS_PROFILE --query 'Endpoints[].{Name:EndpointName,Status:EndpointStatus,Created:CreationTime}' --output table 2>/dev/null || echo "No endpoints found"
        read -p "Enter endpoint name manually: " ENDPOINT_NAME
    fi
fi

if [ -z "$ENDPOINT_NAME" ]; then
    echo "❌ No endpoint name provided. Exiting."
    exit 1
fi

echo "🎯 Using SageMaker endpoint: $ENDPOINT_NAME"

# Function to check if endpoint exists and get status
check_endpoint_status() {
    echo "📊 Checking endpoint status..."
    if aws sagemaker describe-endpoint --endpoint-name "$ENDPOINT_NAME" --profile $AWS_PROFILE >/dev/null 2>&1; then
        aws sagemaker describe-endpoint --endpoint-name "$ENDPOINT_NAME" --profile $AWS_PROFILE \
            --query '{
                EndpointName: EndpointName,
                EndpointStatus: EndpointStatus,
                CreationTime: CreationTime,
                LastModifiedTime: LastModifiedTime,
                FailureReason: FailureReason,
                ProductionVariants: ProductionVariants[0].{
                    VariantName: VariantName,
                    CurrentInstanceCount: CurrentInstanceCount,
                    CurrentWeight: CurrentWeight,
                    DeployedModelName: DeployedModelName
                }
            }' --output table
        return 0
    else
        echo "❌ Endpoint not found: $ENDPOINT_NAME"
        return 1
    fi
}

# Function to find all related log groups
find_log_groups() {
    echo "🔍 Finding related log groups..."
    
    # Possible log group patterns
    LOG_PATTERNS=(
        "/aws/sagemaker/Endpoints/$ENDPOINT_NAME"
        "/aws/sagemaker/Endpoints"
        "/aws/sagemaker/TrainingJobs"
        "/aws/sagemaker/ProcessingJobs"
    )
    
    for pattern in "${LOG_PATTERNS[@]}"; do
        echo "  Checking pattern: $pattern"
        GROUPS=$(aws logs describe-log-groups --log-group-name-prefix "$pattern" --profile $AWS_PROFILE --query 'logGroups[].logGroupName' --output text 2>/dev/null || echo "")
        if [ -n "$GROUPS" ]; then
            echo "  ✅ Found: $GROUPS"
        else
            echo "  ❌ No groups found for pattern: $pattern"
        fi
    done
    
    echo ""
    echo "📋 All SageMaker log groups:"
    aws logs describe-log-groups --log-group-name-prefix "/aws/sagemaker" --profile $AWS_PROFILE \
        --query 'logGroups[].{LogGroup:logGroupName,Created:creationTime,Size:storedBytes}' --output table 2>/dev/null || echo "No SageMaker log groups found"
}

# Function to view recent endpoint logs
view_recent_logs() {
    local time_period=${1:-"1h"}
    echo "📋 Recent endpoint logs (last $time_period):"
    
    LOG_GROUP="/aws/sagemaker/Endpoints/$ENDPOINT_NAME"
    
    if aws logs describe-log-groups --log-group-name-prefix "$LOG_GROUP" --profile $AWS_PROFILE --query 'logGroups[0]' >/dev/null 2>&1; then
        echo "✅ Found log group: $LOG_GROUP"
        echo "📜 Recent logs:"
        aws logs tail "$LOG_GROUP" --since $time_period --profile $AWS_PROFILE --format short
    else
        echo "❌ Log group not found: $LOG_GROUP"
        echo "🔍 Searching for alternative log groups..."
        
        # Try to find any logs with the endpoint name
        aws logs describe-log-groups --profile $AWS_PROFILE --query "logGroups[?contains(logGroupName, '$ENDPOINT_NAME')].logGroupName" --output text
    fi
}

# Function to tail logs in real-time
tail_logs() {
    LOG_GROUP="/aws/sagemaker/Endpoints/$ENDPOINT_NAME"
    echo "📡 Tailing logs from: $LOG_GROUP"
    echo "   (Press Ctrl+C to stop)"
    
    if aws logs describe-log-groups --log-group-name-prefix "$LOG_GROUP" --profile $AWS_PROFILE >/dev/null 2>&1; then
        aws logs tail "$LOG_GROUP" --follow --profile $AWS_PROFILE
    else
        echo "❌ Cannot tail - log group not found: $LOG_GROUP"
    fi
}

# Function to search for errors in logs
search_errors() {
    local hours=${1:-"24"}
    echo "❌ Searching for errors in the last $hours hours:"
    
    LOG_GROUP="/aws/sagemaker/Endpoints/$ENDPOINT_NAME"
    START_TIME=$(date -u -d "$hours hours ago" +%s)000
    
    echo "🔍 Searching in log group: $LOG_GROUP"
    echo "⏰ Start time: $(date -u -d "$hours hours ago")"
    
    # Search for various error patterns
    ERROR_PATTERNS=("ERROR" "error" "Error" "FAILED" "failed" "Failed" "Exception" "Traceback")
    
    for pattern in "${ERROR_PATTERNS[@]}"; do
        echo "🔎 Searching for pattern: $pattern"
        aws logs filter-log-events \
            --log-group-name "$LOG_GROUP" \
            --profile $AWS_PROFILE \
            --start-time $START_TIME \
            --filter-pattern "$pattern" \
            --query 'events[].{Time:timestamp,Message:message}' \
            --output table 2>/dev/null || echo "  No logs found for pattern: $pattern"
    done
}

# Function to view CloudWatch metrics
view_metrics() {
    echo "📈 SageMaker Endpoint CloudWatch Metrics:"
    
    local start_time=$(date -u -d '2 hours ago' --iso-8601)
    local end_time=$(date -u --iso-8601)
    
    echo "⏰ Time range: $start_time to $end_time"
    
    # Invocations
    echo ""
    echo "🔢 Invocations:"
    aws cloudwatch get-metric-statistics \
        --namespace "AWS/SageMaker" \
        --metric-name "Invocations" \
        --dimensions Name=EndpointName,Value="$ENDPOINT_NAME" \
        --start-time "$start_time" \
        --end-time "$end_time" \
        --period 300 \
        --statistics Sum \
        --profile $AWS_PROFILE \
        --query 'Datapoints[].{Time:Timestamp,Count:Sum}' \
        --output table 2>/dev/null || echo "No invocation metrics found"
    
    # Model Latency
    echo ""
    echo "⏱️  Model Latency (ms):"
    aws cloudwatch get-metric-statistics \
        --namespace "AWS/SageMaker" \
        --metric-name "ModelLatency" \
        --dimensions Name=EndpointName,Value="$ENDPOINT_NAME" \
        --start-time "$start_time" \
        --end-time "$end_time" \
        --period 300 \
        --statistics Average,Maximum \
        --profile $AWS_PROFILE \
        --query 'Datapoints[].{Time:Timestamp,AvgLatency:Average,MaxLatency:Maximum}' \
        --output table 2>/dev/null || echo "No latency metrics found"
    
    # 4XX Errors
    echo ""
    echo "❌ 4XX Errors:"
    aws cloudwatch get-metric-statistics \
        --namespace "AWS/SageMaker" \
        --metric-name "Invocation4XXErrors" \
        --dimensions Name=EndpointName,Value="$ENDPOINT_NAME" \
        --start-time "$start_time" \
        --end-time "$end_time" \
        --period 300 \
        --statistics Sum \
        --profile $AWS_PROFILE \
        --query 'Datapoints[].{Time:Timestamp,Errors:Sum}' \
        --output table 2>/dev/null || echo "No 4XX error metrics found"
    
    # 5XX Errors
    echo ""
    echo "💥 5XX Errors:"
    aws cloudwatch get-metric-statistics \
        --namespace "AWS/SageMaker" \
        --metric-name "Invocation5XXErrors" \
        --dimensions Name=EndpointName,Value="$ENDPOINT_NAME" \
        --start-time "$start_time" \
        --end-time "$end_time" \
        --period 300 \
        --statistics Sum \
        --profile $AWS_PROFILE \
        --query 'Datapoints[].{Time:Timestamp,Errors:Sum}' \
        --output table 2>/dev/null || echo "No 5XX error metrics found"
}

# Function to test the endpoint
test_endpoint() {
    echo "🧪 Testing SageMaker endpoint..."
    
    # Create a minimal test payload
    cat > /tmp/test_payload.json << 'EOF'
{
    "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    "method": "tesseract"
}
EOF
    
    echo "📤 Sending test request to endpoint..."
    echo "🎯 Endpoint: $ENDPOINT_NAME"
    echo "📦 Payload: $(cat /tmp/test_payload.json)"
    
    if aws sagemaker-runtime invoke-endpoint \
        --endpoint-name "$ENDPOINT_NAME" \
        --content-type "application/json" \
        --accept "application/json" \
        --body fileb:///tmp/test_payload.json \
        --profile $AWS_PROFILE \
        /tmp/response.json 2>/dev/null; then
        
        echo "✅ Endpoint invocation successful!"
        echo "📥 Response:"
        if [ -f /tmp/response.json ]; then
            cat /tmp/response.json | jq . 2>/dev/null || cat /tmp/response.json
        fi
    else
        echo "❌ Endpoint invocation failed!"
        echo "🔍 Check the error above and endpoint status"
    fi
    
    # Cleanup
    rm -f /tmp/test_payload.json /tmp/response.json
}

# Function to show menu
show_menu() {
    echo ""
    echo "🔍 SageMaker Endpoint Logs & Monitoring Menu:"
    echo "1. Check endpoint status"
    echo "2. Find all log groups"
    echo "3. View recent logs (last 1 hour)"
    echo "4. View recent logs (last 10 minutes)"
    echo "5. Tail logs (live stream)"
    echo "6. Search for errors (last 24 hours)"
    echo "7. Search for errors (last 4 hours)"
    echo "8. View CloudWatch metrics"
    echo "9. Test endpoint with sample request"
    echo "10. Exit"
    echo ""
}

# Main execution
echo "🎯 SageMaker Endpoint Logs Viewer"
echo "📡 Endpoint: $ENDPOINT_NAME"
echo "🔧 AWS Profile: $AWS_PROFILE"

# Check if endpoint exists
if ! check_endpoint_status; then
    echo "❌ Cannot proceed without valid endpoint"
    exit 1
fi

# Main menu loop
while true; do
    show_menu
    read -p "Choose an option (1-10): " choice
    
    case $choice in
        1)
            check_endpoint_status
            ;;
        2)
            find_log_groups
            ;;
        3)
            view_recent_logs "1h"
            ;;
        4)
            view_recent_logs "10m"
            ;;
        5)
            tail_logs
            ;;
        6)
            search_errors "24"
            ;;
        7)
            search_errors "4"
            ;;
        8)
            view_metrics
            ;;
        9)
            test_endpoint
            ;;
        10)
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "❌ Invalid option. Please choose 1-10."
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
done
