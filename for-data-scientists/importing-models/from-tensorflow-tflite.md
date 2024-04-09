# From TensorFlow / TFLite

## About Tensorflow

![](../../.gitbook/assets/TensorFlow\_logo.svg.png)

**TensorFlow** is a [free and open-source](https://en.wikipedia.org/wiki/Free\_and\_open-source\_software) [software library](https://en.wikipedia.org/wiki/Library\_\(computing\)) for [machine learning](https://en.wikipedia.org/wiki/Machine\_learning) and [artificial intelligence](https://en.wikipedia.org/wiki/Artificial\_intelligence). It can be used across a range of tasks but has a particular focus on [training](https://en.wikipedia.org/wiki/Types\_of\_artificial\_neural\_networks#Training) and [inference](https://en.wikipedia.org/wiki/Statistical\_inference) of [deep neural networks](https://en.wikipedia.org/wiki/Deep\_neural\_networks).[\[4\]](https://en.wikipedia.org/wiki/TensorFlow#cite\_note-4)[\[5\]](https://en.wikipedia.org/wiki/TensorFlow#cite\_note-YoutubeClip-5)

TensorFlow was developed by the [Google Brain](https://en.wikipedia.org/wiki/Google\_Brain) team for internal [Google](https://en.wikipedia.org/wiki/Google) use in research and production.[\[6\]](https://en.wikipedia.org/wiki/TensorFlow#cite\_note-6)[\[7\]](https://en.wikipedia.org/wiki/TensorFlow#cite\_note-7)[\[8\]](https://en.wikipedia.org/wiki/TensorFlow#cite\_note-8) The initial version was released under the [Apache License 2.0](https://en.wikipedia.org/wiki/Apache\_License\_2.0) in 2015.[\[1\]](https://en.wikipedia.org/wiki/TensorFlow#cite\_note-Credits-1)[\[9\]](https://en.wikipedia.org/wiki/TensorFlow#cite\_note-Metz-Nov9-9)Google released the updated version of TensorFlow, named TensorFlow 2.0, in September 2019.[\[10\]](https://en.wikipedia.org/wiki/TensorFlow#cite\_note-:12-10)

TensorFlow can be used in a wide variety of programming languages, including [Python](https://en.wikipedia.org/wiki/Python\_\(programming\_language\)), [JavaScript](https://en.wikipedia.org/wiki/JavaScript), [C++](https://en.wikipedia.org/wiki/C%2B%2B), and [Java](https://en.wikipedia.org/wiki/Java\_\(programming\_language\)).[\[11\]](https://en.wikipedia.org/wiki/TensorFlow#cite\_note-:13-11) This flexibility lends itself to a range of applications in many different sectors.

## Model deployment from Tensorflow

Model deployment from Tensorflow (or the subset of tensorflow optimized for edge devices called [tflite](https://www.tensorflow.org/lite)) is simple to achieve by exporting your Tensorflow model to ONNX and subsequently using (if neccesary) the `sclblonnx` package to [clean and check the resulting graph](../onnx-requirements.md#automatic-checking-using-the-sclblonnx-check-function) for an upload to the Nx AI cloud.&#x20;

* Exporting from TF to ONNX is easy using [the tf2onnx tools](https://github.com/onnx/tensorflow-onnx), as illustrated in these examples: [here](https://github.com/scailable/sclblonnx/blob/master/examples/example\_04.py) and [here](https://onnxruntime.ai/docs/tutorials/tf-get-started.html#converting-a-model).
* Otherwise, you can upload the TFLite model directly, and the cloud will take care of exporting the model to ONNX.

After obtaining a [clean ONNX graph that adheres to our requirements](../onnx-requirements.md), you can upload it to the [Nx AI cloud for deployment](../../nx-ai-cloud/upload-your-model.md).
