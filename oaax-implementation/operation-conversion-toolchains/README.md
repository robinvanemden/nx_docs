# Operation Conversion Toolchains

The conversion toolchain is employed to transform a trained non-optimized AI model through a sequence of micro conversion steps, culminating in the creation of a highly optimized model that leverages the target hardware architecture to its fullest potential.\
Note that a conversion toolchain has the capability to generate multiple optimized model files, each tailored to a specific version of the hardware. Additionally, it may decline to optimize a model if the target hardware does not support a particular model.

Those conversion steps may include:

* **Model Pruning**: Pruning involves removing unnecessary connections or parameters from the model without significantly impacting its performance. This reduces the model's size and computational requirements, making it more efficient during inference.
* **Quantization**: Quantization involves reducing the precision of the model's weights and activations, typically from floating-point numbers to lower precision fixed-point numbers (e.g., 8-bit integers). This reduces memory usage and computational complexity, leading to faster inference.
* **Model Compression**: Techniques such as knowledge distillation or model distillation involve training a smaller model to mimic the behavior of a larger, more complex model. This smaller model can then be used for inference, offering a trade-off between model size and performance.
* **Hardware-specific Optimization**: Tailor the model to leverage specific hardware accelerators like CPUs, GPUs, NPUs, or specialized inference chips.

For example, Nvidia utilizes [TensorRT](https://docs.nvidia.com/tao/tao-toolkit/text/trtexec\_integration/index.html)™ to optimize an ONNX model, generating a serialized TensorRT engines from models that is subsequently employed by the runtime to execute inference tasks efficiently.\
Similarly, Hailo provides an SDK called the [Dataflow Compiler](https://hailo.ai/products/hailo-software/hailo-ai-software-suite/#sw-dc) to compile ONNX models to a HEF files used by their [HailoRT](https://hailo.ai/products/hailo-software/hailo-ai-software-suite/#sw-hailort) runtime.

