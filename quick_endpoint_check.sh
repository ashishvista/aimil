#!/bin/bash
# quick_endpoint_check.sh - Quick commands to check SageMaker endpoint status

AWS_PROFILE="test-prod"

echo "🎯 Quick SageMaker Endpoint Check"

# Get endpoint name from Terraform
ENDPOINT_NAME=$(terraform output -raw sagemaker_endpoint_name 2>/dev/null || echo "")

if [ -z "$ENDPOINT_NAME" ]; then
    echo "❌ Could not get endpoint name from Terraform"
    echo "📋 Available endpoints:"
    aws sagemaker list-endpoints --profile $AWS_PROFILE --query 'Endpoints[].{Name:EndpointName,Status:EndpointStatus}' --output table
    exit 1
fi

echo "📡 Endpoint Name: $ENDPOINT_NAME"

# 1. Check endpoint status
echo ""
echo "1️⃣ Endpoint Status:"
aws sagemaker describe-endpoint --endpoint-name "$ENDPOINT_NAME" --profile $AWS_PROFILE \
    --query '{Status:EndpointStatus,Created:CreationTime,Modified:LastModifiedTime,Failure:FailureReason}' \
    --output table

# 2. Check for recent logs
echo ""
echo "2️⃣ Recent Logs (last 30 minutes):"
LOG_GROUP="/aws/sagemaker/Endpoints/$ENDPOINT_NAME"
if aws logs describe-log-groups --log-group-name-prefix "$LOG_GROUP" --profile $AWS_PROFILE >/dev/null 2>&1; then
    echo "✅ Log group found: $LOG_GROUP"
    aws logs tail "$LOG_GROUP" --since 30m --profile $AWS_PROFILE --format short | tail -20
else
    echo "❌ Log group not found: $LOG_GROUP"
    echo "🔍 Available SageMaker log groups:"
    aws logs describe-log-groups --log-group-name-prefix "/aws/sagemaker" --profile $AWS_PROFILE \
        --query 'logGroups[].logGroupName' --output table
fi

# 3. Check for errors in last 2 hours
echo ""
echo "3️⃣ Recent Errors (last 2 hours):"
START_TIME=$(date -u -d '2 hours ago' +%s)000
aws logs filter-log-events \
    --log-group-name "$LOG_GROUP" \
    --profile $AWS_PROFILE \
    --start-time $START_TIME \
    --filter-pattern "ERROR" \
    --query 'events[].{Time:timestamp,Message:message}' \
    --output text 2>/dev/null | head -10 || echo "No errors found"

# 4. Check CloudWatch metrics
echo ""
echo "4️⃣ Recent Metrics (last 1 hour):"
START_TIME=$(date -u -d '1 hour ago' --iso-8601)
END_TIME=$(date -u --iso-8601)

echo "   📊 Invocations:"
aws cloudwatch get-metric-statistics \
    --namespace "AWS/SageMaker" \
    --metric-name "Invocations" \
    --dimensions Name=EndpointName,Value="$ENDPOINT_NAME" \
    --start-time "$START_TIME" \
    --end-time "$END_TIME" \
    --period 300 \
    --statistics Sum \
    --profile $AWS_PROFILE \
    --query 'Datapoints[?Sum > `0`].{Time:Timestamp,Count:Sum}' \
    --output table 2>/dev/null || echo "   No invocation data"

echo "   ⏱️  Average Latency:"
aws cloudwatch get-metric-statistics \
    --namespace "AWS/SageMaker" \
    --metric-name "ModelLatency" \
    --dimensions Name=EndpointName,Value="$ENDPOINT_NAME" \
    --start-time "$START_TIME" \
    --end-time "$END_TIME" \
    --period 300 \
    --statistics Average \
    --profile $AWS_PROFILE \
    --query 'Datapoints[?Average > `0`].{Time:Timestamp,LatencyMs:Average}' \
    --output table 2>/dev/null || echo "   No latency data"

echo ""
echo "🔍 For detailed logs, run: ./view_endpoint_logs.sh"
echo "🧪 To test endpoint, run option 9 in the detailed viewer"
