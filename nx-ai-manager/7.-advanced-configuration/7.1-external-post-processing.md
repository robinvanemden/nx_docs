---
description: >-
  This page describes how to implement an external postprocessor to integrate
  with the Nx AI Manager
---

# 7.1 External Post-processing

It is sometimes desired to add custom or proprietary processing to the inference pipeline. It is therefore possible to add a custom application which receives information from the Edge AI Manager and returns optionally altered information.

Examples are provided in how to create these applications in both C and Python. However these applications can be created using any programming language, as long as the device can execute this program, send it messages over a Unix socket, and receive a response.

A high-level overview of the inference pipeline is as follows:

<figure><img src="../../.gitbook/assets/GeneralFlow.png" alt=""><figcaption><p>A high-level overview of the inference pipeline </p></figcaption></figure>

The postprocessor can therefore receive the inference results from the model, optionally alter these results, and return them. The changes added by the postprocessor will then be sent to the Network Optix platform, where it can be used to generate bounding boxes or events.

Through the settings, instructions can be provided to the Edge AI Manager on how to start the application. The Edge AI Manager will automatically start the applications once necessary, and terminate them once execution is finished.

### External Postprocessor

A postprocessor receives the inference results as a MessagePack encoded buffer. This message is equivalent to what will be sent to the output. The postprocessor can alter this message and return it. The altered message will then be sent to the Network Optix platform. The returned message should have the same structure as the input message, otherwise the Network Optix platform might be unable to parse it. Examples are provided to show how this structure can be parsed, altered, and written.

<figure><img src="../../.gitbook/assets/PPFlow (1).png" alt=""><figcaption><p>External Postprocessor data flow</p></figcaption></figure>

### External Tensor Postprocessor

A setting is provided in which the user can indicate that a postprocessor should receive access to the input tensor which the inference results were generated from. This can be useful for many applications, such as investigating the input tensor within the generated bounding boxes, or even create sub-images.&#x20;

When this setting is enabled, the Edge AI Manager platform will write the input tensor to shared memory, where all external postprocessor can access it. It will then send an additional message to the external postprocessor containing information which can be used to access this shared memory.

The image header message is sent after the inference results message. It is therefore required for the external postprocessor to expect to receive two separate messages before responding with its own message. The image header is also a MessagePack encoded message.

<figure><img src="../../.gitbook/assets/PPTensorFlow (1).png" alt=""><figcaption><p>Tensor Postprocessor data flow</p></figcaption></figure>

The postprocessor can do additional analysis on the tensor data.
