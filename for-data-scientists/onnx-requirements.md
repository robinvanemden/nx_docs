# ONNX requirements

In the previous sections, we have explained how to create ONNX graphs from scratch, providing several examples in the package to help you understand the concept better. However, if you have already trained your model using your preferred training platform or framework, you can simply export it and obtain a file that you can upload to the Nx platform for deployment.

This page offers a comprehensive specification of an ONNX model needed to benefit from all the built-in functionalities in the Nx AI Manager.

## Requirements for the ONNX graph

An ONNX graph/model is an abstract representation of computational graphs that uses nodes to describe operators, such as matrix multiplication, convolutions, and addition.&#x20;

While individuals can construct ONNX models tailored to their specific needs and define new operators or attributes, it's important to note that the Nx AI Manager exclusively accommodates the predefined set of operators outlined in the provided official [specification](https://onnx.ai/onnx/operators/) (also see [here](https://github.com/onnx/onnx/blob/main/docs/Operators.md)). Additionally, each ONNX model includes a property known as "opset\_version," representing the [Operator Set Version](https://github.com/onnx/onnx/blob/main/docs/Versioning.md#operator-sets). It is essential to note that our testing procedures have exclusively evaluated versions up to 17.

Let's talk about the types of ONNX models and variants that can be used with the Nx AI Manager. Currently, the AI Manager supports vision models that process RGB images, but it is not designed to handle textual or any other sensor data without additional coding.

ONNX has a significant advantage in being able to handle graph inputs with dynamic shapes. For example, an image classification model can process images of any shape, such as Height x Width, without prior knowledge of the input shape at runtime. However, it's important to note that our current system only supports graph inputs with static dimensions and does not yet support dynamic input shapes. Within the graph itself, the Nx AI Manager does support dynamic shapes.

To recap, the ONNX models that are supported must:

* be vision models,
* with static input shapes,
* expect to operate on RGB/Grayscale images,
* using only operators defined in this [list](https://onnx.ai/onnx/operators/),&#x20;
* with versions up to 17.

## Named IO specification

The AI Manager streamlines operations by leveraging names of ONNX inputs and outputs (IOs) through built-in functions. This facilitates efficient processing and management within the system.

{% hint style="info" %}
We recommend using [Netron](https://netron.app) to visually inspect your ONNX graph to ensure that the input and outputs match the descriptions provided here.
{% endhint %}

In the following sections, a comprehensive specification will be provided that details the input and output (IO) names of the model, along with their corresponding shapes and data types.

### Inputs specification

It's important to note that while a typical vision model can handle different types of inputs, such as images and confidence scores in different formats and shapes, the AI Manager can only recognize specific, commonly used input types for vision models.&#x20;

The following list gives information about what inputs are supported and how they are expected to be: their names, shapes, and datatypes.

1. **Image input** (_Required_):
   * Description: This input is the primary means of providing images to the model for processing.
   * Name: "`image`"
   * Shape:  The input expects images with two possible shapes: (1, channels, height, width) or (1, height, width, channels), where channels can be either 1 (for Grayscale images) or 3 (for RGB images).
   * Data Type: Floating-point numbers (Float32)
   * Expected value: A tensor containing image values after normalization.
2. **Probability input** (_Optional_):
   * Description: This optional input allows for configuring a threshold score to eliminate bounding boxes with lower detection scores (how likely the bounding box contains an object).&#x20;
   * Name: "`nms_sensitivity`"
   * Shape: (1)
   * Data Type: Floating-point numbers (Float32)
   * Expected value: A floating-point number within the range of \[0, 1].
3. **Mask input** (_Optional_):
   * This input is designed for object detection models and regulates areas in images for model outcomes on a pixel-level granularity.
   * Name: "`mask`"
   * Shape: (Height, Width)
   * Data Type: Boolean (BOOL)
   * Expected value:  A binary value (0 or 1) indicating whether the pixel is included (1) or excluded (0) from the model's processing.

{% hint style="info" %}
While the Mask and Probability inputs are particularly beneficial for object detection models, it's noteworthy that they possess versatile applications within the ONNX graph, extending beyond their immediate utility.
{% endhint %}

### Outputs specification

### Supported ONNX Model Outputs

While ONNX models can produce a wide variety of outputs (scores, bounding boxes, masks, body skeleton, etc.) in numerous formats (number, vectors/list, matrix, etc.), our system currently is able to parse two primary types of outputs that are crucial for most vision-based applications: **Scores** (for image classification models) and **Bounding Boxes** (for object detection models). These outputs are standardized to ensure compatibility and efficiency within our AI Manager framework.&#x20;

Below is an overview of each supported output type:

1. **Scores**:
   * Description: A list of confidence scores for the detected objects.
   * Naming Syntax: "`scores-class_id:class_name`". \
     For example: "scores-0:Cat;1:Dog;2:Horse;3:Nothing".
   * Shape: (1, number of classes)
   * Data Type: Floating-point numbers (Float32)
   * Expected value: Each element of the list is interpreted as a probability/score value.
   * _**Usability**_: These scores are used to visualize the model output by picking the category with the highest score.
2. **Alarms**:
   * Description: A list of boolean values for the detected objects.
   * Naming Syntax: "`alarm-class_id:class_name`". \
     For example: "alarm-0:Cat;1:Dog;2:Horse".
   * Shape: (1, number of classes)
   * Data Type: Boolean values (0 or 1)
   * Expected value: Each element within the list denotes a condition that, when met, triggers an alarm.
   * _**Usability**_: These boolean values serve the purpose of triggering an alarm whenever a True value is present within the list. For instance, if a cat, dog, or horse is detected, an alarm will be raised accordingly.
3. **Bounding Boxes (bboxes)**:
   * Description: Coordinates for the bounding boxes around detected objects represented as a matrix.
   * Naming Syntax: "`bboxes-format:xyxysc;class_id:class_name`"\
     For example: "bboxes-format:xyxysc;0:Person;1:Bike;2:Car".
   * Shape: (Number of detected bboxes, 6)
   * Data Type: Floating-point numbers (Float32)
   * Expected value: Each entry in this list comprises 6 values representing the coordinates of the top-left corner and the bottom-right corner of the bounding box, followed by the model's confidence score, and the class ID (xmin, ymin, xmax, ymax, score, class), thus the format value in the output name being: xyxysc.
   * _**Usability**_: These outputs can be seamlessly integrated with a built-in post-processing mechanism, configurable based on user settings. This post-processing capability extends to various functionalities, including visualization, line-crossing analysis, QR/Barcode scanning, or the application of bounding-box area blurring.

These IO names and formats play a critical role in ensuring the proper functioning of the Nx AI Manager. They are utilized to determine which input values each model input expects during inference and how to interpret the output values generated at the end of inference. This adherence to specified naming conventions and formats enables seamless interaction between the AI Manager and the deployed models, facilitating efficient inference processes.

### Beyond image classification and object detection models

Certain AI models generate raw outputs that require specialized processing, such as image segmentation or pose estimation models. These outputs cannot be directly incorporated into the ONNX model and necessitate hard-coded post-processing using external procedures. This involves extracting insightful information from the model's raw outputs, such as bounding boxes or scores, using specific processing code. The extracted scores or bounding boxes should adhere to the specifications outlined in the previous subsections to ensure compatibility with the Nx AI Manager.
