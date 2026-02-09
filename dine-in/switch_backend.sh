#!/bin/bash
# Switch VLM backend between OVMS and Embedded modes
# Usage: ./switch_backend.sh <backend_type>
# backend_type: "ovms" or "embedded"

set -e

BACKEND_TYPE="$1"

if [ -z "$BACKEND_TYPE" ]; then
    echo "Usage: $0 <backend_type>"
    echo "  backend_type: ovms or embedded"
    exit 1
fi

if [ "$BACKEND_TYPE" != "ovms" ] && [ "$BACKEND_TYPE" != "embedded" ]; then
    echo "Error: backend_type must be 'ovms' or 'embedded'"
    exit 1
fi

echo "==================================================================="
echo "Switching VLM Backend to: ${BACKEND_TYPE^^}"
echo "==================================================================="
echo ""

cd /home/intel/jsaini/order-accuracy

# Update docker-compose.yaml
echo "Updating docker-compose.yaml..."

if [ "$BACKEND_TYPE" = "ovms" ]; then
    # Switch to OVMS backend
    sed -i 's/VLM_BACKEND: embedded/VLM_BACKEND: ovms/' docker-compose.yaml
    echo "✓ Set VLM_BACKEND to ovms"
else
    # Switch to embedded backend
    sed -i 's/VLM_BACKEND: ovms/VLM_BACKEND: embedded/' docker-compose.yaml
    echo "✓ Set VLM_BACKEND to embedded"
fi

# Restart services
echo ""
echo "Restarting services..."
docker compose up -d application-service frame-selector

echo "Waiting for services to start..."
sleep 15

# Verify backend is active
echo ""
echo "Verifying backend..."
BACKEND_LOG=$(docker logs oa_application 2>&1 | grep "VLM Configuration" | tail -1)

if echo "$BACKEND_LOG" | grep -q "backend=${BACKEND_TYPE^^}"; then
    echo "✓ Backend successfully switched to ${BACKEND_TYPE^^}"
    echo ""
    echo "$BACKEND_LOG"
else
    echo "⚠ Warning: Could not verify backend switch"
    echo "Please check logs: docker logs oa_application"
fi

echo ""
echo "Backend switch complete!"
echo ""
echo "Current configuration:"
grep "VLM_BACKEND:" docker-compose.yaml | head -1
echo ""
