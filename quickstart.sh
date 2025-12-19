#!/bin/bash
# Quick start script for cBioPortal Article Classification

set -e

echo "=== cBioPortal Article Classification - Quick Start ==="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
uv sync

# Check if .env exists
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "⚠️  Please edit .env and configure your AWS settings:"
    echo "   - AWS_REGION"
    echo "   - AWS_PROFILE"
    echo "   - BEDROCK_MODEL_ID"
    echo ""
    read -p "Press Enter after configuring .env..."
fi

echo ""
echo "=== Quick Start Complete ==="
echo ""
echo "To use the tool, activate the virtual environment first:"
echo ""
echo "For bash/zsh:"
echo "  source .venv/bin/activate"
echo ""
echo "For fish:"
echo "  source .venv/bin/activate.fish"
echo ""
echo "Then run commands like:"
echo "  cbioportal-classify status"
echo "  cbioportal-classify fetch --download-pdfs --max-downloads 10"
echo "  cbioportal-classify classify --max-papers 5"
echo "  cbioportal-classify analyze"
echo ""
echo "Or run the full pipeline:"
echo "  cbioportal-classify run-all --download-pdfs --max-downloads 10 --max-papers 5"
echo ""
