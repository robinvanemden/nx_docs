# From Edge Impulse

## About Edge Impulse

![](<../../.gitbook/assets/Unknown (1).png>)

_"Edge Impulse is the leading development platform for machine learning on edge devices, free for developers and trusted by enterprises."_ You can find the Edge Impulse training platform [here](https://www.edgeimpulse.com).

## Preliminaries

In this section of our documentation we describe how to use the [Edge Impulse model training platform](https://edgeimpulse.com) to train advanced machine learning models for vision tasks and deploy them seamlessly using the [Nx AI cloud](broken-reference). To follow the documentation at this point we assume that you have access to the following:

* An edge device with the Nx AI manager installed. If you do not have access to an edge device with the Nx AI manager installed please see how to purchase a device. Do make sure you can login to the edge device and navigate to the [AI manager installed on the device](broken-reference).
* A camera that can be used as an input source. Please do see our [documentation regarding camera choice and setup](broken-reference) to get started.
* A Nx AI cloud account. You can register for a free trial account at [https://admin.sclbl.net/register](https://admin.sclbl.net/register). However, the Edge Impulse integration will need to be activated for your account by our support team; the quickest way to get that done is by reaching out [through our chat](https://scailable.net/chat/).
* An Edge Impulse account. Sign up for a free Edge Impulse account at [https://studio.edgeimpulse.com/](https://studio.edgeimpulse.com/).

Once you have all of the above setup, you should be able to proceed to train your own model using edge impulse and deploy it using Nx AI Manager.&#x20;

## Quick overview

We will demonstrate how to train and deploy your own model step-by-step. And, we will show you how to re-train your model once it has been deployed in-the-field. We will cover the following steps:

1. **Model training using the Edge Impulse platform**. Note that we will not provide an elaborate walk through of the amazing capabilities of the Edge Impulse platform; these can be found in the Edge Impulse docs: [https://docs.edgeimpulse.com/docs/](https://docs.edgeimpulse.com/docs/).
2. **Coupling your Edge Impulse model with Nx AI cloud**. We will show how to sync your Edge Impulse model with your model catalog.
3. **Deploying (and testing) your model on your edge device**. This section will detail how to deploy your Edge Impulse model effortlessly to your edge device using the [Nx AI cloud](broken-reference).&#x20;
4. **Retraining your model**. This step is optional, but cool. Once you have a model setup you can collect new training examples in the field and use these to retrain a model. Once done you can iterate (go back to step 1) and get better!

## 1. Model training using the Edge Impulse platform

We start the development of a novel edge AI solution by creating a new project on the Edge Impulse platform:

![](<../../.gitbook/assets/Screen Shot 2023-02-06 at 15.05.48.png>)

The Edge Impulse platform is very intuitive, and allows you to upload and annotate training examples and to train object detection models. We will focus on the Edge Impulse's FOMO model; find a quick getting started guide here: [https://docs.edgeimpulse.com/docs/tutorials/detect-objects-using-fomo](https://docs.edgeimpulse.com/docs/tutorials/detect-objects-using-fomo).

The important bit for this tutorial is to train an object detection model and to select the correct FOMO models. Work through the data acquisition and impulse creation steps in the Edge Impulse platform to get to the object detection model:

![](<../../.gitbook/assets/Screen Shot 2023-02-06 at 15.06.50.png>)

<img src="../../.gitbook/assets/Screen Shot 2023-02-06 at 15.08.48.png" alt="" data-size="original">

Do make sure to select the **FOMO MobileNetV2 (both 0.1 and 0.35), or Yolov5 option**. Next, after you have clicked "Start training" and the model training has finished, you are done (for now) on the Edge Impulse platform.

{% hint style="warning" %}
At this point we **only** support imports of the **FOMO MobileNetV2 and Yolov5** from Edge Impulse. We will be adding support for more Edge Impulse models shortly.
{% endhint %}

## 2. Coupling your Edge Impulse model with Nx

After training your model, you can leave the Edge Impulse platform (but do leave it open in a tab) and move to [https://admin.sclbl.nxvms.com/](https://admin.sclbl.nxvms.com/). After logging in at the Nx AI cloud you will arrive at your dashboard showing your current models and devices (which might both be 0 when you are just getting started):

![](<../../.gitbook/assets/Screenshot from 2024-05-22 13-49-04.png>)

Click the model tab on the left, and next click the  "Add a model" button:

![](<../../.gitbook/assets/Screenshot from 2024-05-22 13-56-45.png>)![](<../../.gitbook/assets/Screenshot from 2024-05-22 13-57-31.png>)

You will arrive at the model upload page, from where you can [select "linking an Edge Impulse project](https://admin.sclbl.net/link-edgeimpulse)". Yes, you can also just click [this](https://admin.sclbl.net/link-edgeimpulse) link. This all should get you here:

![](<../../.gitbook/assets/Screenshot from 2024-05-22 13-57-34.png>)

At this point you can use your Edge Impulse API key and project ID to import your trained model directly from Edge Impulse. Your API key kan be found at your dashboard, and the Project ID can be found in the URL:

![](<../../.gitbook/assets/Screen Shot 2023-02-06 at 15.21.41.png>)

After filling out the API- and project- keys you can click the "Link model" button, and your Edge Impulse model will be imported into your Nx AI library:

![](<../../.gitbook/assets/Screenshot from 2024-05-22 14-26-41.png>)

You can obviously change the model name and documentation (as usual), but effectvely, after the import, the model is directly available for deployment. Once you click "Return to models" you will see the model on the top of you model list:

![](<../../.gitbook/assets/Screenshot from 2024-05-22 14-28-47.png>)

You are now ready to deploy your model to your selected edge device.

## 3. **Deploying (and testing) your model on your edge device**

There are multiple ways in which you can use the Nx AI cloud to [deploy your model to you selected edge device](../../nx-ai-cloud/deployment-and-device-management.md). However, if you still need to configure the edge device, the easiest way of setting things up is to navigate to the AI manager that is running on the device. It can usually be found at port `8081` or through the device setup menu. You should get here:

![](<../../.gitbook/assets/Screen Shot 2023-02-06 at 15.30.57.png>)

At this point you are configuring the setup of the Nx AI manager on this specific device.  For a more elaborate setup please see our [AI manager documentation](broken-reference). However, the steps are simple enough:

1. First, go to the model tab, click the "Select model" button, and select your newly coupled Edge Impulse model from the list. At this point, if you have not yet done so, you might be asked to [register your device](broken-reference).\
   \
   ![](<../../.gitbook/assets/Screen Shot 2023-02-06 at 15.33.25.png>)
2. After [selecting a model](broken-reference), you can select your [input settings](broken-reference). At this point you are able to select and preview up to four input cameras (depending on your targeted edge device).\
   \
   ![](<../../.gitbook/assets/Screen Shot 2023-02-06 at 15.36.09.png>)
3. Finally, you can configure your [output settings](broken-reference). The default settings should work just fine.

Once you are done configuring, you can navigate to the ["Run" tab of the AI manager](broken-reference) to start generating inferences. Alternatively, you can run a single inference [test on the edge device](broken-reference) to see if everything is correctly configured. Testing should give something like this:

![](<../../.gitbook/assets/Screen Shot 2023-02-06 at 15.41.26.png>)\


This all worked. Great.&#x20;

However, at this point it is good to also understand the generated output; here we present the top of the generated JSON that will be send to your specified output location (by default, the Nx data logger):

```
// Example JSON output (top):

[
  {
    "deviceId": "0ec6b6913d844ac1bd1621ddaaf726a0-2",
    "modelId": "31107bd7-b3d2-4f4e-9598-fb46b8e5e4b5",
    "sourceId": "input-0",
    "sourceName": "",
    "outputType": "json",
    "outputFormat": "namedObject",
    "outputDims": [
      [
        1,
        12,
        12,
        3
      ]
    ],
    "outputDataTypes": [
      1
    ],
    "output": {
      "StatefulPartitionedCall:0": [
        0.9991597,
        0.0004198,
        0.0004206,
        0.9988859,
        0.0008452,
        0.0002688,
        0.9887891,
        0.0037218,
```

\
The JSON object starts with some meta data describing the device, model, and camera name. Next, you see the output dimensions. In this case the dimensions are `12 x 12 x 3` which is the standard Edge Impulse FOMO output when a model contains 3 output classes: effectively the model output is a 12x12 grid on top of the 96x96 pixel input image (the image is automatically rescaled by the AI manager) detailing for each of the 12x12=144 blocks of the image which class is detected. What follows is a list (called `StatefulPartitionedCall:0:` of effectively triplets containing the probability for each output class. _I.e.,_ in the above output, the first three blocks of the image are identified as class 1 with probablilities `.9991597`, `.9988859`, and `.9887891` respectively.

You can use the output anyway you want by sending it to the Nx data logger or your own application platform.&#x20;

That's it really; you have just trained and edge deployed a pretty nifty AI model.&#x20;

{% hint style="info" %}
Please note that depending on which option you chose for **Resize mode** when configuring the Impulse during training, you can configure the AI Manager to adopt the same mode by changing the value of [**InputCameraXAspectRatio**](broken-reference) in the settings file.
{% endhint %}

## 4. **Retraining your model**

Although steps 1 to 3 basically got you started, there are a few nice tricks you can use to improve your solution over time. Particularly, you can set the on-device AI manager to capture new training images when needed. On the "Output" tab in the AI manager, you will see the "Upload images with low certainty" box:

![](<../../.gitbook/assets/Screen Shot 2023-02-06 at 15.50.43.png>)

{% hint style="info" %}
Please note that all the functionality described here can also be configured by directly editing the [on-device settings](broken-reference).
{% endhint %}

The image capture feature, which is specific for Edge Impulse models, allows you to set a threshold controlling whether or not an input image will be stored to become input for model re-training. If you set the "Probability threshold" to `.8` for example, _any image that contains one or more block(s) (out of the 144 blocks) for which the highest class probability (i.e., the probability of the recognized class) is lower than `.8` will be send together with the model's output_. By default this image added as a based 64 encoded string to the output JSON:

![](<../../.gitbook/assets/Screen Shot 2023-02-06 at 15.49.53.png>)

After running the model for some time, and assuming you are using default logging of the output data to the Nx data logger, you will be able to view your device in the Nx AI cloud, and view the resulting data:

![](<../../.gitbook/assets/Screen Shot 2023-02-06 at 15.54.48.png>)

Once you have collected a batch of data it is possible to directly upload the data to the Edge Impulse project you started with by clicking the upload to Edge Impulse button:

![](<../../.gitbook/assets/Screen Shot 2023-02-06 at 16.02.03.png>)

At this point you can navigate back to your Edge Impulse project, label the images, retrain the model, and [re-deploy your model](from-edge-impulse.md#3.-deploying-and-testing-you-model-on-your-edge-device).

## Wrap up

The above covers the basics of "training-using-Edge-Impulse-deploying-using-Nx". Very cool stuff, and in this article we really only scratched the surface of the potential applications. If you want to learn more, feel free to [reach out anytime](https://scailable.net/chat/).
