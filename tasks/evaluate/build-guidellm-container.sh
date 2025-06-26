#!/bin/bash

# Build script for GuideLLM UBI container
# Usage: ./build-guidellm-container.sh [REGISTRY] [TAG]

set -e

# Default values
DEFAULT_REGISTRY="quay.io/your-org"
DEFAULT_TAG="guidellm-ubi:latest"

# Parse arguments
REGISTRY=${1:-$DEFAULT_REGISTRY}
TAG=${2:-$DEFAULT_TAG}
FULL_IMAGE="$REGISTRY/$TAG"

echo "🏗️  Building GuideLLM UBI container..."
echo "Registry: $REGISTRY"
echo "Tag: $TAG"
echo "Full image: $FULL_IMAGE"

# Build the container
echo "📦 Building container image..."
podman build -t "$FULL_IMAGE" -f Containerfile .

echo "✅ Build complete!"
echo "Image: $FULL_IMAGE"

# Ask if user wants to push
read -p "Do you want to push this image to the registry? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Pushing image to registry..."
    podman push "$FULL_IMAGE"
    echo "✅ Push complete!"
else
    echo "ℹ️  Image built locally. To push later, run:"
    echo "   podman push $FULL_IMAGE"
fi

echo "🎉 Done!"
echo ""
echo "To test the container locally:"
echo "  podman run --rm -e GUIDELLM_TARGET=http://your-model-url $FULL_IMAGE"
echo ""
echo "To use in OpenShift:"
echo "  Update the image reference in modelcar-guidellm-containerized-task.yaml to: $FULL_IMAGE"