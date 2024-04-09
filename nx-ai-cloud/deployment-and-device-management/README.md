---
description: How do we use the platform for model deployment.
---

# Deployment and device management

The Nx AI Cloud platform allows you to (mass) deploy models to target edge devices. Effectively, you can "swap" the models that run on an edge device (configured using the [AI manager](broken-reference)) remotely and change the device configuration. The latter you can do at a large scale: you can flexibly group devices and deploy models to groups of devices.

## An overview of your devices

The DEVICES tab on the Nx AI Cloud platform shows all devices registered to your account. Device registration is the first step in setting up your AI manager on your edge device, and the result will be a listing of the device in this overview.

<figure><img src="../../.gitbook/assets/image (106).png" alt=""><figcaption></figcaption></figure>

Note that you can manage the device name, view its activity, and view additional details regarding the device installation on this page. You can also group devices for mass deployment.

## AI model deployment and management

In addition to listing your devices, the Nx AI Cloud platform allows you to deploy models to your devices. Although deployment can be done directly from the device, mass deployment using the Platform is easier and more scalable.

To deploy a model to a device, select a model and subsequently select a device or a device group (for mass deployment) to assign the model to the device.&#x20;

To deploy a model to a group of devices, select the group in the device listing and click the "Assign to {groupname}" button.

<figure><img src="../../.gitbook/assets/image (107).png" alt=""><figcaption></figcaption></figure>

In the next step, you will be guided to the models listing, where you can choose a model to assign.

<figure><img src="../../.gitbook/assets/image (108).png" alt=""><figcaption></figcaption></figure>

The blue bar at the top of the page allows you to cancel the assignment.

{% hint style="info" %}
Our edge devices regularly "ping" our servers to inspect model and configuration deployments whenever they can access our cloud. After a model is assigned to a device, the device will download the assigned model at the first opportunity.
{% endhint %}

### Tracking your model assignments

On the cloud, each model detail page displays a listing of the devices where this model is used.

Alternatively, the used model is also shown on the detail page for each device.

<figure><img src="../../.gitbook/assets/image (109).png" alt=""><figcaption></figcaption></figure>



