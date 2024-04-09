# Importing models

Next to creating your own models, it is very common to train AI or ML models using some training platform or tool which are subsequently imported to the Nx AI cloud. In this section of our docs we provide information on how to export models using various model training tools, and we give tips on how to make sure your model can be uploaded to the Nx AI cloud and deployed to the Nx AI manager.

{% hint style="success" %}
Note that in many cases the tools we discuss here allow you to export your trained model to ONNX. Once you have the ONNX graph, please see if its fits our requirements before uploading it to the Nx AI cloud.
{% endhint %}

Below we discuss training models using:

* [Edge Impulse](https://www.edgeimpulse.com). Note that for EdgeImpulse exports it is not necessary to convert to ONNX.
* [Tensorflow / tensorflow lite](from-tensorflow-tflite.md).&#x20;
* [PyTorch](from-pytorch.md).
* [Teachable machine](from-teachable-machine.md).
* Other platforms that allows you to export to ONNX.
