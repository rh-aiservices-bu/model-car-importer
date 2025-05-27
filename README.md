# ModelCar Pipeline for OpenShift

This OpenShift pipeline automates:
- Downloading a model from Hugging Face.
- Optionally compressing the model using GPTQ quantization.
- Storing the model in a Persistent Volume.
- Building a ModelCar OCI image.
- Pushing the image to an OCI registry (e.g., Quay.io).

## 🚀 Prerequisites

Ensure you have:
1. **OpenShift CLI (`oc`) installed** - [Install OpenShift CLI](https://docs.openshift.com/container-platform/latest/cli_reference/openshift_cli/getting-started-cli.html)
2. **OpenShift Pipelines (Tekton) installed**
3. **GPU-enabled nodes** - Required for model compression

---

## Installation

### 1. Create the Required OpenShift Secret

This example is for quay.io

To push images to **Quay.io**, you need to create an **OpenShift Secret** that stores your authentication credentials. Follow these steps to generate and configure the secret:  

#### Log in to Quay.io
- Go to [Quay.io](https://quay.io/) and log in to your account.  

#### Create or Select a Robot Account
- Click on your **profile icon** in the top-right corner and select **Account Settings**.  
- Navigate to the **Robot Accounts** section.  
- Create a new robot account **(or select an existing one)**.  

#### Assign Write Permissions  
- Grant the **robot account** **write permissions** for the repository where you want to push images.  

#### Download the Secret File  
- Click on the **robot account** to open its details.  
- Find the **Kubernetes Secret** option and **download the YAML file**.  

#### Modify and Apply the Secret  
- Open the downloaded YAML file in a text editor.  
- Change the `metadata.name` field to **`quay-auth`**.  
- Apply the secret to OpenShift using:  
  ```sh
  oc apply -f quay-auth-secret.yaml
  ```

#### Verify the Secret  
To confirm the secret has been created, run:  
```sh
oc get secrets | grep quay-auth
```

Now, your OpenShift cluster can authenticate with **Quay.io** for secure image pushes. 🚀

---

### 2. Apply the Tekton Pipeline in OpenShift
Apply the pipeline YAML to your OpenShift project:
```sh
oc apply -f modelcar-pipeline.yaml
```

Verify the pipeline is created:
```sh
tkn pipeline list
```

---

### 3. Apply the modelcar-storage in OpenShift

Apply the pipeline YAML to your OpenShift project:
```sh
oc apply -f modelcar-storage.yaml
```

### 4. Create a secret for your Huggingface token

If the model you are downloading from huggingface requires a token, create a secret with:

```
oc create secret generic huggingface-secret \
  --from-literal=HUGGINGFACE_TOKEN=your_actual_token_here
```

---

## Model Compression

The pipeline includes an optional model compression step using GPTQ quantization. This can significantly reduce the model size while maintaining reasonable performance.

### Compression Details

The compression process:
1. Uses GPTQ (Generative Pre-trained Transformer Quantization) with W4A16 scheme
2. Quantizes linear layers to 4-bit precision while keeping activations in 16-bit
3. Preserves the original model in a backup directory (`model_original`)
4. Automatically calculates and reports size reduction statistics

### Compression Parameters

The compression is configured with the following parameters:
- Group size: 16 (for quantization granularity)
- Calibration samples: 16 per GPU
- Maximum sequence length: 64 (for calibration)
- Memory per GPU: 16GB

### Resource Requirements

Compression requires:
- NVIDIA GPUs (4 GPUs recommended)
- 24GB memory per GPU
- 1 CPU core per GPU

---

## Usage Instructions

Edit the contents of `modelcar-pipelinerun.yaml` to specify:
- `HUGGINGFACE_MODEL`: The model to download from Hugging Face
- `OCI_IMAGE`: The destination OCI image
- `HUGGINGFACE_ALLOW_PATTERNS`: File patterns to download (default: "*.safetensors *.json *.txt")
- `COMPRESS_MODEL`: Set to "true" to enable compression

Example configuration:
```yaml
params:
  - name: HUGGINGFACE_MODEL
    value: "ibm-granite/granite-3.2-2b-instruct"
  - name: OCI_IMAGE
    value: "quay.io/my-user/my-modelcar"
  - name: HUGGINGFACE_ALLOW_PATTERNS
    value: "*.safetensors *.json *.txt"
  - name: COMPRESS_MODEL
    value: "true"
```

Run the pipeline with:
```sh
oc create -f modelcar-pipelinerun.yaml
```

## Checking the pipeline status

```sh
oc get pipelinerun
```

To view detailed logs:
```sh
tkn pipelinerun logs <pipelinerun-name>
```

## Clean up

To remove completed and failed pods, run the following commands:

```sh
oc delete pod --field-selector=status.phase==Succeeded
oc delete pod --field-selector=status.phase==Failed
```