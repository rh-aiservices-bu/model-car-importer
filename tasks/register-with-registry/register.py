#!/usr/bin/env python3
import os
import sys
import subprocess
import json

print("Developed by the Red Hat AI Customer Adoption and Innovation team (CAI)")
if os.environ.get('SKIP_TASK') in os.environ.get('SKIP_TASKS', '').split(','):
    print("Skipping register-with-registry task")
    sys.exit(0)

from model_registry import ModelRegistry
from model_registry.types import ModelArtifact, ModelVersion, RegisteredModel
from model_registry.exceptions import StoreError

# Get namespace from service account
namespace_file_path = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
with open(namespace_file_path, 'r') as namespace_file:
    namespace = namespace_file.read().strip()

# Get cluster domain from environment
cluster_domain = os.environ.get('CLUSTER_DOMAIN', 'cluster.local')

# Get server address from environment
server_address = os.environ.get('MODEL_REGISTRY_URL')
if not server_address:
    print("Error: MODEL_REGISTRY_URL environment variable not set")
    sys.exit(1)
print(f"Server address: {server_address}")

# Set token path in environment
os.environ["KF_PIPELINES_SA_TOKEN_PATH"] = "/var/run/secrets/kubernetes.io/serviceaccount/token"
print("Token path set in environment")

# Initialize model registry client
registry = ModelRegistry(
    server_address=server_address,
    port=443,
    author="modelcar-pipeline",
    is_secure=False
)

# Extract model name from Hugging Face model ID
model_name = os.environ.get('MODEL_NAME').lower()
model_version = os.environ.get('MODEL_VERSION')
oci_image = "oci://" + os.environ.get('OCI_IMAGE').lower() + ":latest"

try:
    # Register model with metadata
    registered_model = registry.register_model(
        model_name,
        oci_image,
        version=model_version,
        description=f"Model downloaded from Hugging Face and compressed using GPTQ",
        model_format_name="safetensors",
        model_format_version="1",
        storage_key="modelcar-storage",
        storage_path="/workspace/shared-workspace/model",
        metadata={
            "source": "huggingface",
            "framework": "pytorch",
            "compressed": True
        }
    )

    print(f"Successfully registered model: {model_name} version {model_version}")
    print(f"Model URI: {oci_image}")
    print(f"Model details available at: https://rhods-dashboard-redhat-ods-applications.{cluster_domain}/modelRegistry/modelcar-pipeline-registry/registeredModels/1/versions/{registry.get_model_version(model_name, model_version).id}/details")

    # Save model version ID for deployment task
    model_version_id = registry.get_model_version(model_name, model_version).id
    with open('/workspace/shared-workspace/model_version_id', 'w') as f:
        f.write(str(model_version_id))

except StoreError as e:
    print(f"Model version already exists: {model_name} version {model_version}")
    try:
        # Get existing model version details
        model_details = registry.get_model_version(model_name, model_version)
        print(f"Existing model details:")
        print(f"Model ID: {model_details.id}")
        print(f"Model Name: {model_details.name}")
        
        # Save model version ID for deployment task
        with open('/workspace/shared-workspace/model_version_id', 'w') as f:
            f.write(str(model_details.id))
        
        print("Task completed successfully - using existing model version")
        sys.exit(0)
    except Exception as e:
        print(f"Error getting existing model details: {str(e)}")
        sys.exit(1)
except Exception as e:
    print(f"Error registering model: {str(e)}")
    sys.exit(1) 