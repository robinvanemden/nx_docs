# Common Models

The Nx toolkit currently offers support for image classification and object detection models. To facilitate easy deployment, we provide several tutorials containing instructions on how to deploy common models such as MobileNet, ResNet, EfficientNet, ViT, LeViT for image classification, and YoloS, Yolov4, Yolov7, Yolov8 for object detection.

Key steps involved in this process include:

1. Incorporating necessary post-processing steps such as masking and a configurable Non-Maximum Suppression (NMS) for object detection models and Softmax for image classification models directly into the ONNX graph.
2. Modifying the model's input and output shapes and names as per the requirements of the AI Manager.
3. Making sure the exported ONNX has an Operator Set Version less than 18.

The primary objective of these tutorials is to equip AI developers with off-the-shelf scripts, enabling them to deploy these models within minutes. This is achieved by exporting the models to ONNX format, ensuring compatibility with the AI Manager as described [previously](../../onnx-requirements.md).

{% hint style="info" %}
All Python and Shell scripts mentioned in the tutorials can be found [here](https://drive.google.com/file/d/17cbft9gy9Li6tAx4nyx8XQ\_teS7EnT4i/view?usp=sharing).
{% endhint %}
