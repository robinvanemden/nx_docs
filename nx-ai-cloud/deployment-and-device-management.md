---
description: How do we use the platform for model deployment.
---

# Deployment and device management

The Nx AI Cloud platform allows you to (mass) deploy models to target devices. Effectively, you can "swap" the models that run on a device (configured using the [AI manager](broken-reference)) remotely and change the device configuration. The latter you can do at a large scale: you can deploy models to all the devices on a server.

## An overview of your systems and devices

The SYSTEMS tab on the Nx AI Cloud platform shows all systems registered to your cloud account.&#x20;

If the system is on-online the name of the system and the Details button are available.

<figure><img src="../.gitbook/assets/Screenshot from 2024-05-22 13-32-48.png" alt=""><figcaption><p>overview of systems with one on-line system</p></figcaption></figure>

If the system is off-line you cannot continue until the system comes on-line.

<figure><img src="../.gitbook/assets/Screenshot from 2024-05-22 13-33-39.png" alt=""><figcaption><p>listing of one system that is off-line</p></figcaption></figure>

## Overview of a single system

If you select an on-line system you are directed to the system page.

<figure><img src="../.gitbook/assets/Screenshot from 2024-05-22 13-34-42.png" alt=""><figcaption><p>overview of a single on-line system</p></figcaption></figure>

The page shows the servers in the system and lists all the devices and their groups.

The same page when the system is off-line. You cannot do anything now.

<figure><img src="../.gitbook/assets/Screenshot from 2024-05-22 13-35-29.png" alt=""><figcaption><p>overview of a single off-line system</p></figcaption></figure>

## AI model deployment and management

You can assign or replace a model on a single server or you can select multiple servers and assign a single model to all the selected servers.

### Assign or replace a model on a single server

To assign or replace a model on a single server you need to click on the 'Details' button on the system page. You will be directed to the server page.

<figure><img src="../.gitbook/assets/Screenshot from 2024-05-22 13-36-57.png" alt=""><figcaption><p>overview of a single server in the system</p></figcaption></figure>

To assign an additional model to the server you can press 'Assign a new model' or the 'add' button. To replace a model press the button with the left-right arrow in it.

You will be directed to the 'Models listing'.

On the top you get a new navigation bar that shows you are currently in assigning mode. You can use that to go back to the models listing page, back to the server or cancel the whole assignment process.

Once you have found the model you want to assign or replace use the button 'Use this model'.&#x20;

<figure><img src="../.gitbook/assets/Screenshot from 2024-05-22 13-38-32.png" alt=""><figcaption><p>models listing</p></figcaption></figure>

Now, one of two things could have happened.

One: You do not have to re-authenticate yourself and are redirected back to the server page directly with a message that the model has been assigned.

<figure><img src="../.gitbook/assets/success-model-assign.png" alt=""><figcaption></figcaption></figure>

Two: You get a pop-up where you need to confirm your credentials. This is normal when you want to change some things in an Nx system and have logged in a while ago.

<figure><img src="../.gitbook/assets/Screen Shot 2024-04-23 at 10.34.41.png" alt=""><figcaption><p>re-authentocation pop-up</p></figcaption></figure>

You need to enter your credentials. When that is done, the pop-up is closed and you need to press the button 'Use this model' again. You are then redirected back to the server page.

### Assign a single model to all the selected servers

To start you need to be on the system page, which shows you all the servers and all devices in the system.

Just below the 'Servers' header on this page there is a link called 'here' which starts the process of selecting server to assign a model to.

<figure><img src="../.gitbook/assets/Screenshot from 2024-05-22 13-41-21.png" alt=""><figcaption></figcaption></figure>

Once pressed, a new column with checkboxes will be added to the servers and you can use them to select the servers you want to assign a new model to.

<figure><img src="../.gitbook/assets/Screenshot from 2024-05-22 13-40-10.png" alt=""><figcaption><p>multiple server selection activated</p></figcaption></figure>

If you select one or more servers the 'Assign a new model' will activate. Pressing this button will start the model assignment just like updating a model for a single server.

Note: assigning a model to multiple servers will always remove all other models and replace them with the new model you selected.
