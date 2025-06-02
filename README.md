# ModelCar Pipeline

A Tekton pipeline for downloading models from Hugging Face, compressing them, packaging them into ModelCar images, and deploying them on OpenShift AI along with a deployment of Anything LLM.

![ModelCar Pipeline](assets/pipeline.png)

## Prerequisites

- OpenShift AI cluster with GPU-enabled node (e.g., AWS EC2 g6.12xlarge instance providing 4 x NVIDIA L4 Tensor Core GPUs)
- Access to Quay.io (for pushing images)
- Access to Hugging Face (for downloading models)
- OpenShift model registry service
- Service account with appropriate permissions
- Quay.io authentication secret

## Features

- Downloads models from Hugging Face with customizable file patterns
- Optional model compression using RHAIIS LLM Compressor
- Packages models into OCI images using [OLOT](https://github.com/containers/olot)
- Pushes images to Quay.io
- Registers models in the OpenShift model registry
- Deployment as InferenceService with GPU support
- Waits until the model is deployed to complete pipeline
- Deploys AnythingLLM UI configured to use the deployed model
- Supports skipping specific tasks


## Deployment Steps

### 1. Create Required Namespace

```bash
# Create a new namespace for the pipeline
oc new-project modelcar-pipeline
```

### 2. Create Required Secrets

#### Create a Quay.io Robot Account

1. Log in to your Quay.io account
2. Navigate to your organization or user account
3. Go to "Robot Accounts" in the left sidebar
4. Click "Create Robot Account"
5. Give the robot account a name (e.g., `modelcar-robot`)
6. Select the repository you want to push to
7. Grant "Write" permissions to the repository
8. Click "Create Robot Account"
9. Save the generated username and password (you'll only see the password once)

#### Create the Quay.io Secret

Set the robot account credentials as environment variables:

```bash
# Set your Quay.io robot account credentials
export QUAY_USERNAME="ROBOT_USERNAME"
export QUAY_PASSWORD="ROBOT_PASSWORD"
```

Create a Kubernetes secret with the robot account credentials:

```bash
# Create the secret using the robot account credentials
cat <<EOF | oc create -f -
apiVersion: v1
kind: Secret
metadata:
  name: quay-auth
type: kubernetes.io/dockerconfigjson
data:
  .dockerconfigjson: $(echo -n '{"auths":{"quay.io":{"auth":"'$(echo -n "${QUAY_USERNAME}:${QUAY_PASSWORD}" | base64)'"}}}' | base64)
EOF
```

#### Create Hugging Face Token Secret

Login to Huggingface https://huggingface.co/settings/tokens and copy your access_token

Set this as an environment variable with 

```bash
export HF_TOKEN=xxx
```

Create Hugging Face token secret by running:

```bash
cat <<EOF | oc create -f -
apiVersion: v1
kind: Secret
metadata:
  name: huggingface-secret
type: Opaque
data:
  HUGGINGFACE_TOKEN: $(echo $HF_TOKEN | base64)
EOF
```

# Set the Model Registry URL environment variable:

```bash
# Get the model registry URL from your OpenShift AI cluster

# Create an environment variable to store the model registry url.
export MODEL_REGISTRY_URL="https://model-registry.apps.yourcluster.com"
```

### 3. Create Service Account and Permissions

```bash
# Create service account
oc create serviceaccount modelcar-pipeline

# Create role for model registry access
cat <<EOF | oc create -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: model-registry-access
rules:
- apiGroups: ["serving.kserve.io"]
  resources: ["inferenceservices", "servingruntimes"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["get", "list"]
EOF

# Bind role to service account
cat <<EOF | oc create -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: model-registry-access
subjects:
- kind: ServiceAccount
  name: modelcar-pipeline
roleRef:
  kind: Role
  name: model-registry-access
  apiGroup: rbac.authorization.k8s.io
EOF
```

### 4. Update Resource Quota to ensure 4 gpus are allowed

```bash
# Create or update resource quota for GPU resources
cat <<EOF | oc apply -f -
apiVersion: v1
kind: ResourceQuota
metadata:
  name: modelcar-pipeline-core-resource-limits
spec:
  hard:
    requests.nvidia.com/gpu: "4"
    limits.nvidia.com/gpu: "4"
EOF
```

### 5. Create Storage

```bash
# Create storage class for pipeline workspace
oc create -f modelcar-storage.yaml
```

### 6. Deploy Pipeline

Create the pipeline
```bash
oc create -f modelcar-pipeline.yaml
```

Create the compress-task
```bash
oc create -f modelcar-compress-task.yaml
```

Create the compress-script configmap from the Python file which contains the python code to run the LLM Compression.

The `compress.py` script:
- Uses the LLM Compressor library to compress the model using GPTQ quantization
- Configures compression parameters like bits (4-bit quantization) and group size
- Handles multi-GPU compression for faster processing
- Saves the compressed model in the same format as the original
- Includes progress tracking and error handling

```bash
oc create configmap compress-script --from-file=compress.py
```

# Create the pipeline run

Set your Quay.io repository:

```bash
# Set your Quay.io repository (replace with your repository)
export QUAY_REPOSITORY="quay.io/your-org/your-repo"
```

This example will pull and compress the `ibm-granite/granite-3.2-2b-instruct` model.  You can change this to use other models from huggingface you have access to.

```bash
cat <<EOF | oc create -f -
apiVersion: tekton.dev/v1beta1
kind: PipelineRun
metadata:
  name: modelcar-pipelinerun
spec:
  pipelineRef:
    name: modelcar-pipeline
  serviceAccountName: modelcar-pipeline
  params:
    - name: HUGGINGFACE_MODEL
      value: "ibm-granite/granite-3.2-2b-instruct"
    - name: OCI_IMAGE
      value: "${QUAY_REPOSITORY}"
    - name: HUGGINGFACE_ALLOW_PATTERNS
      value: "*.safetensors *.json *.txt"
    - name: COMPRESS_MODEL
      value: "true"
    - name: MODEL_NAME
      value: "granite-3.2-2b-instruct"
    - name: MODEL_VERSION
      value: "1.0.0"
    - name: MODEL_REGISTRY_URL
      value: "${MODEL_REGISTRY_URL}"
    - name: DEPLOY_MODEL
      value: "true"
  workspaces:
    - name: shared-workspace
      volumeClaimTemplate:
        spec:
          accessModes:
            - ReadWriteOnce
          resources:
            requests:
              storage: 50Gi
    - name: quay-auth-workspace
      secret:
        secretName: quay-auth
  podTemplate:
    tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
    nodeSelector:
      nvidia.com/gpu.present: "true"
EOF
```

### 7. Verify Deployment

```bash
# Check pipeline status
oc get pipelinerun

```

## Pipeline Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `HUGGINGFACE_MODEL` | Hugging Face model repository (e.g., "ibm-granite/granite-3.2-2b-instruct") | - |
| `OCI_IMAGE` | OCI image destination (e.g., "quay.io/my-user/my-modelcar") | - |
| `HUGGINGFACE_ALLOW_PATTERNS` | Space-separated list of file patterns to allow (e.g., "*.safetensors *.json *.txt") | "" |
| `COMPRESS_MODEL` | Whether to compress the model using GPTQ (true/false) | "false" |
| `MODEL_NAME` | Name of the model to register in the model registry | - |
| `MODEL_VERSION` | Version of the model to register | "1.0.0" |
| `SKIP_TASKS` | Comma-separated list of tasks to skip | "" |
| `MODEL_REGISTRY_URL` | URL of the model registry service | - |
| `DEPLOY_MODEL` | Whether to deploy the model as an InferenceService (true/false) | "false" |

### Skipping Tasks

The pipeline supports skipping specific tasks using the `SKIP_TASKS` parameter. This is useful for example if you want to deploy a model without redoing the entire pipeline. For example, to skip all tasks up to the deploy stage:

```bash
SKIP_TASKS="cleanup-workspace,pull-model-from-huggingface,compress-model,build-and-push-modelcar,register-with-registry"
```

## Model Deployment

When `DEPLOY_MODEL` is set to "true", the pipeline will:
1. Create a ServingRuntime with GPU support
2. Deploy an InferenceService using the model
3. Wait for the service to be ready
4. Save the service URL to the workspace
5. Deploy AnythingLLM UI configured to use the deployed model

The deployment includes:
- GPU resource allocation
- Memory and CPU limits
- Automatic scaling configuration
- Service URL detection
- Health monitoring
- AnythingLLM UI with:
  - Generic OpenAI-compatible endpoint configuration


### AnythingLLM Configuration

The AnythingLLM UI is automatically configured with:
- Connection to the deployed model via generic OpenAI-compatible endpoint

The UI is accessible via a secure HTTPS route with edge termination.


## Monitoring

To monitor the pipeline execution:

```bash
# Check pipeline status
oc get pipelinerun modelcar-pipelinerun

# Check InferenceService status (if deployed)
oc get inferenceservice

```

## Notes

- Model compression is optional and can be skipped
- The pipeline supports skipping specific tasks using the `SKIP_TASKS` parameter
- Model deployment requires GPU-enabled nodes in the cluster
- The service URL is saved to the workspace for future reference

## Testing the Model

Once the pipeline completes successfully, you can access the AnythingLLM UI to test your model:

1. Get the AnythingLLM route:
```bash
oc get route anything-llm -o jsonpath='{.spec.host}'
```

2. Open the URL in your browser (it will be in the format `https://anything-llm-<namespace>.<cluster-domain>`)

3. In the AnythingLLM UI:
   - The model is pre-configured to use your deployed model
   - You can start a new chat to test the model's responses
   - The UI provides a user-friendly interface for interacting with your model

4. To verify the model is working correctly:
   - Try sending a simple prompt like "Hello, how are you?"
   - Check that the response is generated in a reasonable time
   - Verify that the responses are coherent and relevant