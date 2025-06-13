# ModelCar Builder Container

This container is used in the ModelCar pipeline to build and push model images using OLOT.

## Building the Container

To build the container image using Podman:

```bash
# Navigate to the directory containing the Containerfile
cd tasks/build-and-push-modelcar

# Build the container image
podman build --platform linux/amd64  -t quay.io/hayesphilip/modelcar-builder:latest -f Containerfile .
```

## Pushing to Quay.io

Before pushing, make sure you're logged in to Quay.io:

```bash
# Login to Quay.io
podman login quay.io
```

Then push the image:

```bash
# Push the image to Quay.io
podman push quay.io/hayesphilip/modelcar-builder:latest
```

## Container Contents

The container includes:
- Python 3.10
- Poetry for Python package management
- Git for cloning OLOT
- Skopeo for OCI image operations
- OLOT for model packaging

## Usage in Pipeline

This container is used in the ModelCar pipeline to:
1. Clone and install OLOT
2. Download the base OCI image
3. Package model files into the OCI image
4. Push the final image to Quay.io

## Development

To modify the container:
1. Update the Containerfile as needed
2. Rebuild the image using the commands above
3. Push the new image to Quay.io
4. Update the pipeline to use the new image if necessary 