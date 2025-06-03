# ModelCar Registry Container

This container is used in the ModelCar pipeline to register models with the OpenShift model registry.

## Building the Container

To build the container image using Podman:

```bash
# Navigate to the directory containing the Containerfile
cd tasks/register-with-registry

# Build the container image for linux/amd64 platform
podman build --platform linux/amd64 -t quay.io/hayesphilip/modelcar-register:latest -f Containerfile .
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
podman push quay.io/hayesphilip/modelcar-register:latest
```

## Container Contents

The container includes:
- Python 3.10
- model-registry==0.2.15 package
- curl for HTTP requests

## Usage in Pipeline

This container is used in the ModelCar pipeline to:
1. Register models with the OpenShift model registry
2. Handle model versioning
3. Store model metadata
4. Link models to their OCI images

The pipeline uses the registration script from the ConfigMap to:
1. Connect to the model registry
2. Register new models or update existing ones
3. Store model metadata and version information
4. Save model version IDs for deployment

## Development

To modify the container:
1. Update the Containerfile as needed
2. Rebuild the image using the commands above
3. Push the new image to Quay.io
4. Update the pipeline to use the new image if necessary

To modify the registration script:
1. Update register.py as needed
2. Recreate the ConfigMap using the command in the main README
3. The pipeline will automatically use the updated script 