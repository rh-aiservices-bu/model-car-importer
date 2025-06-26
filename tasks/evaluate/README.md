# GuideLLM UBI Container

This directory contains a Red Hat Universal Base Image (UBI) based Containerfile for building a GuideLLM performance benchmarking container.

## Overview

The container is built using Red Hat UBI9 Python 3.9 base image and includes:
- GuideLLM performance benchmarking tool
- Environment variable configuration support
- Proper Red Hat labeling and metadata
- Non-root user execution for security

## Building the Container

### Prerequisites
- Podman or Docker
- Access to Red Hat UBI registry (registry.access.redhat.com)
- Access to your target container registry (e.g., Quay.io)

### Build Script Usage

```bash
# Make the build script executable
chmod +x build-guidellm-container.sh

# Build with default settings
./build-guidellm-container.sh

# Build with custom registry and tag
./build-guidellm-container.sh quay.io/your-org guidellm-ubi:v1.0
```

### Manual Build

```bash
# Build the container
podman build -t quay.io/your-org/guidellm-ubi:latest -f Containerfile .

# Push to registry
podman push quay.io/your-org/guidellm-ubi:latest
```

## Environment Variables

The container supports the following GuideLLM configuration variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GUIDELLM_TARGET` | `http://localhost:8000` | Target model server URL |
| `GUIDELLM_MODEL` | `neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16` | Model name |
| `GUIDELLM_RATE_TYPE` | `sweep` | Rate testing type (sweep, fixed) |
| `GUIDELLM_DATA` | `prompt_tokens=256,output_tokens=128` | Token configuration |
| `GUIDELLM_MAX_REQUESTS` | `100` | Maximum number of requests |
| `GUIDELLM_MAX_SECONDS` | _(empty)_ | Maximum test duration |
| `GUIDELLM_OUTPUT_PATH` | `/results/results.json` | Output file path |

## Usage Examples

### Local Testing

```bash
# Test against a local model server
podman run --rm \
  -e GUIDELLM_TARGET=http://localhost:8000/v1 \
  -e GUIDELLM_MODEL=your-model-name \
  -v $(pwd)/results:/results \
  quay.io/your-org/guidellm-ubi:latest
```

### OpenShift Integration

Update the `modelcar-guidellm-containerized-task.yaml` to use your custom image:

```yaml
steps:
  - name: guidellm-evaluate
    image: quay.io/your-org/guidellm-ubi:latest
    # ... rest of configuration
```

### Custom Command Execution

```bash
# Run custom guidellm commands
podman run --rm \
  quay.io/your-org/guidellm-ubi:latest \
  benchmark --target http://your-server/v1 --help
```

## Security Features

- **Non-root execution**: Runs as `guidellm` user (UID 1001)
- **UBI base**: Built on Red Hat certified base images
- **Minimal attack surface**: Only includes necessary components
- **Red Hat compliance**: Follows Red Hat container standards

## Integration with ModelCar Pipeline

This container is designed to work with the ModelCar pipeline's GuideLLM evaluation task:

1. **Automatic service discovery**: Finds deployed InferenceService URLs
2. **Environment configuration**: Sets appropriate GuideLLM variables
3. **Results collection**: Saves benchmark results to shared workspace
4. **Error handling**: Provides clear failure modes and diagnostics

## Troubleshooting

### Build Issues

If you encounter build issues:

```bash
# Check UBI registry access
podman pull registry.access.redhat.com/ubi9/python-39:latest

# Verify git access (for cloning GuideLLM)
git clone https://github.com/neuralmagic/guidellm.git /tmp/test-clone
```

### Runtime Issues

For runtime problems:

```bash
# Check container logs
podman logs <container-id>

# Test connectivity to target server
podman run --rm quay.io/your-org/guidellm-ubi:latest \
  curl -f http://your-server/v1/models
```

### OpenShift Issues

Common OpenShift deployment issues:

1. **Image pull permissions**: Ensure your OpenShift cluster can pull from your registry
2. **Network policies**: Verify the container can reach the target model server
3. **Resource limits**: Check that CPU/memory limits are sufficient

## Contributing

When modifying the Containerfile:

1. Test locally before pushing
2. Update version labels appropriately
3. Maintain Red Hat compliance standards
4. Test integration with ModelCar pipeline

## License

This container definition is part of the ModelCar project and follows the same licensing terms.