#!/bin/bash

#
# OVMS Model Setup Script for Order Accuracy
# This script sets up the OVMS model files needed for the VLM backend
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models_vlm"
SOURCE_MODELS_DIR="/home/intel/jsaini/ovms-vlm/models_vlm"

echo "=========================================="
echo "OVMS Model Setup for Order Accuracy"
echo "=========================================="
echo ""

# Check if source models exist
if [ -d "${SOURCE_MODELS_DIR}/Qwen" ]; then
    echo "✓ Found existing OVMS models at ${SOURCE_MODELS_DIR}"
    echo ""
    echo "Copying models to order-accuracy project..."
    
    # Create Qwen directory if it doesn't exist
    mkdir -p "${MODELS_DIR}/Qwen"
    
    # Copy the model files
    if [ -d "${SOURCE_MODELS_DIR}/Qwen/Qwen2-VL-2B-Instruct" ]; then
        echo "  → Copying Qwen2-VL-2B-Instruct model..."
        cp -r "${SOURCE_MODELS_DIR}/Qwen/Qwen2-VL-2B-Instruct" "${MODELS_DIR}/Qwen/"
        echo "  ✓ Model copied successfully"
    else
        echo "  ✗ Qwen2-VL-2B-Instruct model not found in source directory"
        exit 1
    fi
    
    echo ""
    echo "✓ Model setup complete!"
    echo ""
    echo "Model location: ${MODELS_DIR}/Qwen/Qwen2-VL-2B-Instruct"
    echo "Model size: $(du -sh ${MODELS_DIR}/Qwen/Qwen2-VL-2B-Instruct | cut -f1)"
    echo ""
    
else
    echo "✗ No pre-exported models found at ${SOURCE_MODELS_DIR}"
    echo ""
    echo "You have two options:"
    echo ""
    echo "1. Export the model from HuggingFace:"
    echo "   cd ${SCRIPT_DIR}"
    echo "   pip install -r export_requirements.txt"
    echo "   python export_model.py text_generation \\"
    echo "     --source_model Qwen/Qwen2-VL-2B-Instruct \\"
    echo "     --weight-format int4 \\"
    echo "     --target_device GPU \\"
    echo "     --model_repository_path models_vlm"
    echo ""
    echo "2. Copy from another location:"
    echo "   cp -r /path/to/models_vlm/Qwen ${MODELS_DIR}/"
    echo ""
    exit 1
fi

echo "Next steps:"
echo "1. Start the OVMS service:"
echo "   cd $(dirname ${SCRIPT_DIR})"
echo "   docker-compose --profile ovms up -d"
echo ""
echo "2. Verify OVMS is running:"
echo "   curl http://localhost:8001/v1/config"
echo ""
echo "3. Check model status:"
echo "   curl http://localhost:8001/v1/models"
echo ""
