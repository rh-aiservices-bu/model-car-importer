# ModelCar Pipeline

A Tekton pipeline for downloading models from Hugging Face, compressing them, running evaluation, packaging them into ModelCar images, and deploying them on OpenShift AI along with a deployment of Anything LLM.

![ModelCar Pipeline](assets/pipeline.png)

## Features

- Downloads models from Hugging Face with customizable file patterns
- Optional model compression using RHAIIS LLM Compressor
- Runs evaluation using defined eval tasks e.g. gsm8k against original and compressed model, and ouputs results of compressed model.
- Packages models into OCI images using [OLOT](https://github.com/containers/olot)
- Pushes images to Quay.io
- Registers models in the OpenShift model registry
- Deployment as InferenceService with GPU support
- Waits until the model is deployed to complete pipeline
- Deploys AnythingLLM UI configured to use the deployed model
- Supports skipping specific tasks


## Prerequisites

- OpenShift AI cluster with GPU-enabled node (e.g., AWS EC2 g6.12xlarge instance providing 4 x NVIDIA L4 Tensor Core GPUs)
- Access to Quay.io (for pushing images)
- Access to Hugging Face (for downloading models)
- OpenShift model registry service
- OpenShift CLI (oc)

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```bash
# Quay.io credentials
QUAY_USERNAME="ROBOT_USERNAME"
QUAY_PASSWORD="ROBOT_PASSWORD"
QUAY_REPOSITORY="quay.io/your-org/your-repo"

# Hugging Face token
HUGGINGFACE_MODEL="meta-llama/Llama-3.3-70B-Instruct"
HF_TOKEN="your_huggingface_token"

# Model Registry
MODEL_REGISTRY_URL="https://model-registry.apps.yourcluster.com"

# Model details
MODEL_NAME="Llama-3.3-70B-Instruct"
MODEL_VERSION="1.0.0"
```

You can get your Hugging Face token from [Hugging Face Settings](https://huggingface.co/settings/tokens).

### Creating a Quay.io Robot Account

To create a robot account in Quay.io:

1. Log in to [Quay.io](https://quay.io)
2. Navigate to your organization or user account
3. Click on "Robot Accounts" in the left sidebar menu
4. Click "Create Robot Account" button
5. Enter a name for the robot account (e.g. `modelcar-pipeline`)
6. Click "Create Robot Account"
7. On the next screen, click "Edit Repository Permissions"
8. Search for and select your target repository
9. Set permissions to "Write" access
10. Click "Update Permission"
11. Save the robot account credentials:
    - Username will be in format: `your-org+robot-name`
    - Password will be shown only once - copy it immediately
12. Use these credentials in your `.env` file:
    ```bash
    QUAY_USERNAME="your-org+robot-name"
    QUAY_PASSWORD="robot-account-password" 
    ```

Note: Make sure to save the password when it's displayed as it cannot be retrieved later. 

Before running any commands, source the environment variables:

```bash
# Source the environment variables
source .env

# Verify the variables are set
echo "Using Quay repository: $QUAY_REPOSITORY"
echo "Using model: $MODEL_NAME"
```

## Deployment Steps

### 1. Create Required Namespace

```bash
# Create a new namespace for the pipeline
oc new-project modelcar-pipeline
```

### 2. Create Required Secrets

#### Create the Quay.io Secret

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

### 3. Create Service Account and Permissions

```bash
# Create service account
oc create serviceaccount modelcar-pipeline
```

### 4. Create OpenShift objects

First, create a dynamic resource quota file based on the current project name:

```bash
# Get the current project name
export PROJECT_NAME=$(oc project -q)

# Create the resource quota file
cat <<EOF > openshift/resource-quota.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ${PROJECT_NAME}-core-resource-limits
spec:
  hard:
    requests.cpu: "8"
    requests.memory: 24Gi
    limits.cpu: "16"
    limits.memory: 32Gi
    requests.nvidia.com/gpu: "4"
    limits.nvidia.com/gpu: "4"
EOF

# Create all OpenShift objects
oc apply -f openshift/
```

### 5. Create ConfigMaps

Create the compress-script configmap from the Python file which contains the python code to run the LLM Compression.

The `tasks/compress/compress.py` script:
- Uses the LLM Compressor library to compress the model using GPTQ quantization
- Configures compression parameters like bits (4-bit quantization) and group size
- Handles multi-GPU compression for faster processing
- Saves the compressed model in the same format as the original
- Includes progress tracking and error handling

```bash
oc create configmap compress-script --from-file=tasks/compress/compress.py
```

Create the registration script ConfigMap:

```bash
# Create the ConfigMap from the Python script
oc create configmap register-script --from-file=tasks/register-with-registry/register.py
```

Create the evaluation script ConfigMap:

```bash
# Create the ConfigMap from the evaluation script
oc create configmap evaluate-script --from-file=evaluate.py=tasks/evaluate/evaluate-script.py
```

You can create different evaluation script ConfigMaps for different types of evaluations:

```bash
# Create a script for code evaluation
oc create configmap code-evaluate-script --from-file=evaluate.py=tasks/evaluate/code-evaluate-script.py

```

Then specify which script to use in your PipelineRun:

```bash

    - name: EVALUATION_SCRIPT
      value: "code-evaluate-script"  # Use the code evaluation script

```

### 6. Create PipelineRun

Create the PipelineRun using environment variables:

```bash
cat <<EOF | oc create -f -
apiVersion: tekton.dev/v1beta1
kind: PipelineRun
metadata:
  name: modelcar-pipelinerun
spec:
  pipelineRef:
    name: modelcar-pipeline
  timeout: 3h  # Add a 3-hour timeout
  serviceAccountName: modelcar-pipeline
  params:
    - name: HUGGINGFACE_MODEL
      value: "${HUGGINGFACE_MODEL}"
    - name: OCI_IMAGE
      value: "${QUAY_REPOSITORY}"
    - name: HUGGINGFACE_ALLOW_PATTERNS
      value: "*.safetensors *.json *.txt *.md *.model"
    - name: COMPRESS_MODEL
      value: "true"
    - name: MODEL_NAME
      value: "${MODEL_NAME}"
    - name: MODEL_VERSION
      value: "${MODEL_VERSION}"
    - name: MODEL_REGISTRY_URL
      value: "${MODEL_REGISTRY_URL}"
    - name: DEPLOY_MODEL
      value: "true"
    - name: EVALUATE_MODEL
      value: "true"
    - name: EVALUATION_SCRIPT
      value: "evaluate-script"  # Use the standard evaluation script
  workspaces:
    - name: shared-workspace
      persistentVolumeClaim:
        claimName: modelcar-storage
    - name: quay-auth-workspace
      secret:
        secretName: quay-auth
  podTemplate:
    securityContext:
      runAsUser: 1001
      fsGroup: 1001
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
| `EVALUATE_MODEL` | Whether to evaluate the model using lm-evaluation-harness (true/false) | "false" |
| `MODEL_NAME` | Name of the model to register in the model registry | - |
| `MODEL_VERSION` | Version of the model to register | "1.0.0" |
| `SKIP_TASKS` | Comma-separated list of tasks to skip | "" |
| `MODEL_REGISTRY_URL` | URL of the model registry service | - |
| `DEPLOY_MODEL` | Whether to deploy the model as an InferenceService (true/false) | "false" |
| `EVALUATION_SCRIPT` | Name of the ConfigMap containing the evaluation script | "evaluate-script" |

### Model Evaluation

When `EVALUATE_MODEL` is set to "true", the pipeline will:
1. Install vllm and lm-evaluation-harness
2. Run evaluation using benchmarks defined in the evaluate-script.oy
4. Output evaluation metrics

The evaluation task uses the same GPU resources as the compression task to ensure consistent performance.

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

## Cleanup

To remove all objects created by the pipeline and clean up the namespace, run the following commands:

```bash
# Source environment variables if not already done

PROJECT_NAME=$(oc project -q)

oc delete pipelinerun modelcar-pipelinerun


oc delete -f openshift/

oc delete configmap compress-script
oc delete configmap register-script
oc delete configmap evaluate-script


oc delete secret quay-auth
oc delete secret huggingface-secret


oc delete serviceaccount modelcar-pipeline


oc delete inferenceservice --all --namespace $PROJECT_NAME


oc delete servingruntime --all  --namespace $PROJECT_NAME


oc delete deployment anything-llm


oc delete project $PROJECT_NAME
```


## Other scenarios:

### Deploy an existing model from Quay.io without downloading or compressing:

```bash
cat <<EOF | oc create -f -
apiVersion: tekton.dev/v1beta1
kind: PipelineRun
metadata:
  name: modelcar-deploy-only
spec:
  pipelineRef:
    name: modelcar-pipeline
  timeout: 1h
  serviceAccountName: modelcar-pipeline
  params:
    - name: HUGGINGFACE_MODEL
      value: "${HUGGINGFACE_MODEL}"
    - name: OCI_IMAGE
      value: "${QUAY_REPOSITORY}"
    - name: HUGGINGFACE_ALLOW_PATTERNS
      value: "*.safetensors *.json *.txt *.md *.model"
    - name: COMPRESS_MODEL
      value: "false"
    - name: MODEL_NAME
      value: "${MODEL_NAME}"
    - name: MODEL_VERSION
      value: "${MODEL_VERSION}"
    - name: MODEL_REGISTRY_URL
      value: "${MODEL_REGISTRY_URL}"
    - name: DEPLOY_MODEL
      value: "true"
    - name: EVALUATE_MODEL
      value: "false"
    - name: SKIP_TASKS
      value: "cleanup-workspace,pull-model-from-huggingface,build-and-push-modelcar,register-with-registry"
  workspaces:
    - name: shared-workspace
      persistentVolumeClaim:
        claimName: modelcar-storage
    - name: quay-auth-workspace
      secret:
        secretName: quay-auth
  podTemplate:
    securityContext:
      runAsUser: 1001
      fsGroup: 1001
    tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
    nodeSelector:
      nvidia.com/gpu.present: "true"
EOF
```

This PipelineRun will:
1. Skip the download, compression, and evaluation tasks
2. Use the existing model image from Quay.io
3. Register the model in the model registry
4. Deploy the model as an InferenceService
5. Deploy the AnythingLLM UI

### Pull a model from Hugging Face and deploy without compression or evaluation:

```bash
cat <<EOF | oc create -f -
apiVersion: tekton.dev/v1beta1
kind: PipelineRun
metadata:
  name: modelcar-pull-and-deploy
spec:
  pipelineRef:
    name: modelcar-pipeline
  timeout: 2h
  serviceAccountName: modelcar-pipeline
  params:
    - name: HUGGINGFACE_MODEL
      value: "${HUGGINGFACE_MODEL}"
    - name: OCI_IMAGE
      value: "${QUAY_REPOSITORY}"
    - name: HUGGINGFACE_ALLOW_PATTERNS
      value: "*.safetensors *.json *.txt *.md *.model"
    - name: COMPRESS_MODEL
      value: "false"
    - name: MODEL_NAME
      value: "${MODEL_NAME}"
    - name: MODEL_VERSION
      value: "${MODEL_VERSION}"
    - name: MODEL_REGISTRY_URL
      value: "${MODEL_REGISTRY_URL}"
    - name: DEPLOY_MODEL
      value: "true"
    - name: EVALUATE_MODEL
      value: "false"
  workspaces:
    - name: shared-workspace
      persistentVolumeClaim:
        claimName: modelcar-storage
    - name: quay-auth-workspace
      secret:
        secretName: quay-auth
  podTemplate:
    securityContext:
      runAsUser: 1001
      fsGroup: 1001
    tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
    nodeSelector:
      nvidia.com/gpu.present: "true"
EOF
```

This PipelineRun will:
1. Download the model from Hugging Face
2. Skip compression and evaluation
3. Build and push the ModelCar image to Quay.io
4. Register the model in the model registry
5. Deploy the model as an InferenceService
6. Deploy the AnythingLLM UI

### Code based evals

Ensure the code-evaluate-script configmap is created.
```bash
oc create configmap compress-script --from-file=compress.py=tasks/compress/compress-code.py
```

```bash
cat <<EOF | oc create -f -
apiVersion: tekton.dev/v1beta1
kind: PipelineRun
metadata:
  name: modelcar-coding-python-aggressive
spec:
  pipelineRef:
    name: modelcar-pipeline
  timeout: 6h
  serviceAccountName: modelcar-pipeline
  params:
    - name: HUGGINGFACE_MODEL
      value: "${HUGGINGFACE_MODEL}"
    - name: OCI_IMAGE
      value: "${QUAY_REPOSITORY}"
    - name: HUGGINGFACE_ALLOW_PATTERNS
      value: "*.safetensors *.json *.txt *.md *.model"
    - name: COMPRESS_MODEL
      value: "true"
    - name: MODEL_NAME
      value: "${MODEL_NAME}"
    - name: MODEL_VERSION
      value: "${MODEL_VERSION}"
    - name: MODEL_REGISTRY_URL
      value: "${MODEL_REGISTRY_URL}"
    - name: DEPLOY_MODEL
      value: "true"
    - name: EVALUATE_MODEL
      value: "true"
    - name: MAX_MODEL_LEN
      value: "16000"
    - name: SKIP_TASKS
      value: "cleanup-workspace,pull-model-from-huggingface"
  workspaces:
    - name: shared-workspace
      persistentVolumeClaim:
        claimName: modelcar-storage
    - name: quay-auth-workspace
      secret:
        secretName: quay-auth
  podTemplate:
    securityContext:
      runAsUser: 1001
      fsGroup: 1001
    tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
    nodeSelector:
      nvidia.com/gpu.present: "true"
EOF
```