
# ModelCar Pipeline for OpenShift

This OpenShift pipeline automates:
- Downloading a model from Hugging Face.
- Storing the model in a Persistent Volume.
- Building a ModelCar OCI image.
- Pushing the image to an OCI registry (e.g., Quay.io).

## 🚀 Prerequisites

Ensure you have:
1. **OpenShift CLI (`oc`) installed** - [Install OpenShift CLI](https://docs.openshift.com/container-platform/latest/cli_reference/openshift_cli/getting-started-cli.html)
2. **OpenShift Pipelines (Tekton) installed**

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

---

## Usage Instructions

Edit the contents of modelcar-pipelinerun.yaml to specify HUGGINGFACE_MODEL you wish to download and the OCI_IMAGE you are pushing the modelcal image to.

Run the pipeline with:

```sh
oc create -f modelcar-pipelinerun.yaml

```

## Checking the pipeline status

```sh
oc get pipelinerun
```

## Clean up

To remove completed and failed pods, run the following commands:

```sh
oc delete pod --field-selector=status.phase==Succeeded
oc delete pod --field-selector=status.phase==Failed
```