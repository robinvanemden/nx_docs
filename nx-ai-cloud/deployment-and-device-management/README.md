---
description: How do we use the platform for model deployment.
---

# Deployment and device management

The Nx AI Cloud platform allows you to (mass) deploy models to target devices. Effectively, you can "swap" the models that run on a device (configured using the [AI manager](broken-reference)) remotely and change the device configuration. The latter you can do at a large scale: you can deploy models to all the devices on a server.

## An overview of your systems and devices

The SYSTEMS tab on the Nx AI Cloud platform shows all systems registered to your cloud account.&#x20;

If the system is on-online the name of the system and the Details button are available.

<figure><img src="../../.gitbook/assets/Screen Shot 2024-04-23 at 09.31.10.png" alt=""><figcaption><p>overview of systems with one on-line system</p></figcaption></figure>

If the system is off-line you cannot continue until the system comes on-line.

<figure><img src="../../.gitbook/assets/Screenshot 2024-04-23 at 09.33.41.png" alt=""><figcaption><p>listing of one system that is off-line</p></figcaption></figure>

## Overview of a single system

If you select an on-line system you are directed to the system page.

<figure><img src="../../.gitbook/assets/Screen Shot 2024-04-23 at 09.31.17.png" alt=""><figcaption><p>overview of a single on-line system</p></figcaption></figure>

The page shows the servers in the system and lists all the devices and their groups.

If the system is on-line. You can use the "here" link to select multiple servers and assign a model to all selected servers.

The same page when the system is off-line. You cannot do anything now.

<figure><img src="../../.gitbook/assets/Screen Shot 2024-04-23 at 09.36.38.png" alt=""><figcaption><p>overview of a single off-line system</p></figcaption></figure>

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



