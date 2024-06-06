---
description: Create a loop with a training platform and the Nx AI Manager.
---

# 8.3 Training Loop

## General preparation

This tutorial assumes you have the following things already set up:

* Nx Cloud account.
* Nx Meta system installed.
  * System connected to Nx Cloud account.
  * Client installed.
  * Device added to configuration.
  * Nx AI plugin installed.
* Installed the sclbl-integration-sdk repository.
  * `git clone https://github.com/scailable/sclbl-integration-sdk.git`
  * Followed the ‘Get started’ instructions.



### Edge Impulse preparation

When building a loop with Edge Impulse there are a couple of requirements.

1. Project with a model configured.
2. API key:
   1. From the main dashboard page find the tab ‘Keys’ and select it.
   2. If there are no keys press ‘Add new API key’.
   3. Enter a name and select the role ‘Admin’.
3. Project ID:
   1. The project ID is located in the URL of the page. For instance in\
      https://studio.edgeimpulse.com/studio/1234 the project-id is 1234.\


## Steps

We will describe the loop in 6 steps.

1. Add model to the Nx AI Cloud
2. Update postprocessor
3. Configure model and postprocessor
4. Run the model
5. Check the data
6. Update the model in the cloud
7. Update model in the system



### Step 1: Add model to Nx AI Cloud

We will add the model of the training platform to the Nx AI Cloud.

#### For Edge Impulse

1. Log in the Nx AI Cloud and click ‘Add a model’.&#x20;
2. Select ‘Edge Impulse’, enter the API key and project ID.&#x20;
3. Click ‘Link model’.

Now the model will be downloaded from Edge Impulse and will be prepared for use.\


### Step 2: Update postprocessor

We will take an example postprocessor from the sclbl-integration-sdk and modify it with a specific condition to filter only specific images to send.

#### For Edge Impulse

1. Open the sclbl-integration-sdk into your favourite editor.
2. Open the following Python script: postprocessor-python-edgeimpulse-example/postprocessor-python-edgeimpulse-example.py
3. Find the location where the API is set and replace it with the API-key from your project. Look for the line: `edgeimpulse.API_KEY`
4. Find the line `upload_sample = False` and modify the code below that to suit your condition.
5. Compile and install the postprocessor using the instructions in the README.md of the project.



### Step 3: Configure model and postprocessor

We will select the model and postprocessor.

1. In the Nx client go to the device, right click and select ‘Camera settings’.
2. Go to the tab ‘Plugins’ and select the ‘Nx AI Manager’.
3. Click ‘Manage server’ and click the icon with left-right arrows called ‘Replace model’ or ‘Assign a new model’ if a model was not already assigned.
4. Choose the model from step 1 and click ‘Ok’.
5. Under the heading ‘Assigned Models’, select ‘Post Processor’ and select your post processor from step 2. If you didn’t rename it from the README it is called ‘Example-Postprocessor’.
6. Under the heading ‘Runtime Status’, click ‘Start’ and wait for the status to be ‘Runtime is running’.
7. Click ‘Ok’ at the bottom.\


### Step 4: Run the model

To run the model the device must be visible in the layout.

1. Double click your device to make it visible.
2. On the top right click the left icon ‘Object search’ if the model contains bounding boxes to see them.\


### Step 5: Check the data

Now we are going to check the incoming data, label it and then retrain.

#### Edge Impulse

1. Go to the project page.
2. Select ‘Data acquisition’.
3. Training and Test should contain data coming from the server.
4. Label the data.
5. Select ‘Retrain model’ and then ‘Train model’.



### Step 6: Update the model in the cloud

We are going to update the model in the Nx AI Cloud.

1. Two ways to update the model
   1. Go back to the ‘Manage server’ in the Nx Client / AI plugin and click the name of the model.
   2. Log into the AI Cloud and go to the model.
2. Click the button ‘Update from \<TrainingPlatform>’

\


### Step 7: Update model in the system (WorkInProgress)

We are going to update the models in the system.

1. (option not available yet) In the Nx Client / AI plugin click the button ‘Update models’.
2. Manually delete the model cache. Follow these instructions:
   1. On the cmdline:
      1. `cd /opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/cache`
      2. `rm *`
   2. In the Nx Client / AI plugin stop and start the plugin
