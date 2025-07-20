#!/bin/bash

# Test AWS Profile Methods
# This script demonstrates why --profile flag is more reliable than AWS_PROFILE environment variable

echo "🧪 Testing AWS Profile Methods..."

AWS_PROFILE_NAME="test-prod"

echo ""
echo "📋 Method Comparison:"
echo "===================="

# Method 1: Environment Variable
echo "1️⃣  Testing export AWS_PROFILE method:"
export AWS_PROFILE=$AWS_PROFILE_NAME
aws sts get-caller-identity 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Environment variable method works"
else
    echo "❌ Environment variable method failed"
fi

# Method 2: --profile Flag
echo ""
echo "2️⃣  Testing --profile flag method:"
unset AWS_PROFILE
aws sts get-caller-identity --profile $AWS_PROFILE_NAME 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ --profile flag method works"
else
    echo "❌ --profile flag method failed"
fi

echo ""
echo "🔍 Detailed Analysis:"
echo "===================="

# Check AWS config
echo "📄 AWS Config file (~/.aws/config):"
if [ -f ~/.aws/config ]; then
    echo "Found config file"
    grep -A 5 "\[profile $AWS_PROFILE_NAME\]" ~/.aws/config 2>/dev/null || echo "Profile $AWS_PROFILE_NAME not found in config"
else
    echo "❌ Config file not found"
fi

echo ""
echo "🔐 AWS Credentials file (~/.aws/credentials):"
if [ -f ~/.aws/credentials ]; then
    echo "Found credentials file"
    grep -A 3 "\[$AWS_PROFILE_NAME\]" ~/.aws/credentials 2>/dev/null || echo "Profile $AWS_PROFILE_NAME not found in credentials"
else
    echo "❌ Credentials file not found"
fi

echo ""
echo "🌍 Current Environment Variables:"
echo "AWS_PROFILE: ${AWS_PROFILE:-'(not set)'}"
echo "AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID:-'(not set)'}"
echo "AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY:+'***set***':-'(not set)'}"
echo "AWS_SESSION_TOKEN: ${AWS_SESSION_TOKEN:+'***set***':-'(not set)'}"

echo ""
echo "💡 Recommendation:"
echo "=================="
echo "✅ Always use --profile flag for AWS SSO profiles"
echo "✅ It's more explicit and reliable"
echo "❌ Avoid export AWS_PROFILE with SSO profiles"

echo ""
echo "🔧 To fix login issues:"
echo "aws sso login --profile $AWS_PROFILE_NAME"
