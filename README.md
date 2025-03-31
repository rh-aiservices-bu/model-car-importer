
# **OpenShift Tekton Pipeline: HuggingFace -> ModelCar OCI Image Builder**

This OpenShift Tekton pipeline automates the process of:
1. Downloading a model from **Hugging Face**.
2. Preparing and packaging it as a **ModelCar**.
3. Pushing the image to an image registry securely using authentication from an **OpenShift Secret**.

---

## **Prerequisites**
Before using this pipeline, ensure you have:
- An **OpenShift** cluster with **Tekton Pipelines** installed.
- The **`tkn` CLI** installed for interacting with Tekton Pipelines.
- An image registry e.g. quay.io with permissions to push images.

---

## **Installation**

### **1. Create the Required OpenShift Secret**

This example is for quay.io

To push images to **Quay.io**, you need to create an **OpenShift Secret** that stores your authentication credentials. Follow these steps to generate and configure the secret:  

#### Log in to Quay.io**  
- Go to [Quay.io](https://quay.io/) and log in to your account.  

#### Create or Select a Robot Account**  
- Click on your **profile icon** in the top-right corner and select **Account Settings**.  
- Navigate to the **Robot Accounts** section.  
- Create a new robot account **(or select an existing one)**.  

#### Assign Write Permissions**  
- Grant the **robot account** **write permissions** for the repository where you want to push images.  

#### Download the Secret File**  
- Click on the **robot account** to open its details.  
- Find the **Kubernetes Secret** option and **download the YAML file**.  

#### Modify and Apply the Secret**  
- Open the downloaded YAML file in a text editor.  
- Change the `metadata.name` field to **`quay-auth`**.  
- Apply the secret to OpenShift using:  
  ```sh
  oc apply -f quay-auth-secret.yaml
  ```

#### Verify the Secret**  
To confirm the secret has been created, run:  
```sh
oc get secrets | grep quay-auth
```

Now, your OpenShift cluster can authenticate with **Quay.io** for secure image pushes. 🚀

---

### **2. Apply the Tekton Pipeline in OpenShift**
Apply the pipeline YAML to your OpenShift project:
```sh
oc apply -f modelcar-pipeline.yaml
```

Verify the pipeline is created:
```sh
tkn pipeline list
```

---

---

### **3. Apply the modelcar-storage in OpenShift**

Apply the pipeline YAML to your OpenShift project:
```sh
oc apply -f modelcar-storage.yaml
```

---

## **Usage Instructions**

### **Start the Pipeline**
You can manually trigger the pipeline using the following command (replace quay.io/hayesphilip/modelcar:latest with the quay.io repository)

```sh
tkn pipeline start modelcar-pipeline \
    -p HUGGINGFACE_MODEL=ibm-granite/granite-3.2-2b-instruct \
    -p OCI_IMAGE=quay.io/hayesphilip/modelcar:latest \
    -w name=shared-workspace,claimName=modelcar-storage \
    -w name=quay-auth-workspace,secret=quay-auth
```