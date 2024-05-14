# XPU Operation Runtimes

The second stage in the deployment of AI models in the edge is inference execution carried by the **Runtime** component.

The runtime serves as an interface utilized by software applications requiring inference capabilities. Acting as a bridge between the software application and the underlying hardware, the runtime facilitates the execution of AI models for tasks such as image classification or object detection.\
The software application communicates with the runtime, providing input data and receiving output results, without needing to directly interact with the complexities of model loading, inference execution, or resource management. This abstraction layer offered by the runtime simplifies the integration of machine learning capabilities into software applications, allowing developers to focus on building and enhancing the functionality of their applications without the need for extensive expertise in hardware optimization.
