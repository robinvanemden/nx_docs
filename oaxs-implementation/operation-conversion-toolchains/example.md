# Example

This page provides a concrete example detailing every aspect that needs to be present in conversion toolchain as discussed in the specification page.

In this example, we'll be using Python to create a function that's fed an ONNX file and that produces an ONNX-like file that will be consumed by the Hailo runtime.\
The Python code mainly contains a core function  responsible for the model conversion in the toolchain, and other files useful to Dockerize it.

<details>

<summary>Conversion function</summary>

```python
def convert_to_hailo_onnx(input_path: str, output_path: str, logs):
    logs.add_message('Starting conversion', {'Target Hardware': chosen_hw_arch})
    runner = ClientRunner(hw_arch=chosen_hw_arch)
    try:
        runner.translate_onnx_model(input_path, onnx_model_name,
                                    start_node_names=['tfl.conv_2d'],
                                    end_node_names=['tfl.conv_2d34']
                                    )
    except ParsingWithRecommendationException as e:
        raise Exception(e)
    logs.add_data(**{'Translation': 'Done'})

    io_info = extract_io_info(input_path)
    logs.add_data(**{'Reading IO information': 'Done'})

    runner.optimize(calib_dataset(io_info))
    logs.add_data(**{'Optimizing Model': 'Done'})

    runner.compile()  # the returned HEF is not needed when working with ONNXRT
    logs.add_data(**{'Compiling Model': 'Done'})

    onnx_model = runner.get_hailo_runtime_model()  # only possible on a compiled model
    onnx.save(onnx_model, output_path)  # save model to file
```

</details>

The `convert_to_hailo_onnx` function is the one responsible for parsing, transpiling, optimizing and compiling the input model to build a Hailo-ONNX compatible with the runtime.

The Dockerfile below is used to build a docker image that exposes that function.\
It installs the required Linux and Python dependencies along with the Hailo's Dataflow Compiler toolchain. Finally, in the last line, we expose the `toolchain-run-block` shell script that will call our conversion function.



{% hint style="info" %}
For this example to run, you'll need to download Hailo Dataflow Compiler and HailoRT from [Hailo's Developer Zone](https://hailo.ai/products/hailo-software/hailo-ai-software-suite/#sw-dc).&#x20;

Namely:&#x20;

* hailort-4.16.0-cp38-cp38-linux\_x86\_64.whl
* hailo\_dataflow\_compiler-3.26.0-py3-none-linux\_x86\_64.whl
* hailort\_4.16.0\_amd64.deb

\
Those files need to be included in the `hailo-deps` directory of the full example folder.
{% endhint %}

<details>

<summary>Dockerfile</summary>

```docker
ARG DEBIAN_FRONTEND=noninteractive
FROM ubuntu:22.04

WORKDIR /root

# Update the system and install dependencies
RUN apt-get update && apt-get install -y software-properties-common

# Add the deadsnakes PPA
RUN add-apt-repository ppa:deadsnakes/ppa

# Update the system again and install Python 3.8
RUN apt-get update -y
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y python3.8 python3.8-distutils python3.8-venv python3.8-dev wget \
    python3-tk

RUN wget https://bootstrap.pypa.io/get-pip.py
# Install pip
RUN python3.8 get-pip.py
# Verify the installation
RUN python3.8 --version

RUN echo "## Update and install packages"

RUN echo "### Support packages:"

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get update -qq -y --fix-missing \
  && apt-get install -qq -y gcc g++ curl make xxd wget apt-utils dos2unix software-properties-common git autoconf \
    automake libtool unzip cmake build-essential protobuf-compiler libprotoc-dev graphviz graphviz-dev locales locales-all sudo > /dev/null


COPY requirements.txt /root/requirements.txt
COPY setup.py /root/setup.py
COPY toolchain_block /root/toolchain_block
COPY MANIFEST.in /root/MANIFEST.in
COPY src/run-block.sh /usr/local/bin/toolchain-run-block
COPY hailo-deps/ /root/hailo-deps

RUN echo "### Installing Python dependencies:"
RUN python3.8 -m venv /root/env
ENV PATH="/root/env/bin:$PATH"
RUN pip install -r requirements.txt

ENV PATH="/usr/local/bin:${PATH}"
RUN echo " ###> Copying compile script to dockerfile" \
  && dos2unix /usr/local/bin/toolchain-run-block \
  && chmod +x /usr/local/bin/toolchain-run-block

ENTRYPOINT ["/usr/local/bin/toolchain-run-block"]
```



</details>

{% hint style="success" %}
The full example of the conversion block can be found [here](https://drive.google.com/file/d/1bbUdddjrrkUMpFQfD\_w2DR7UI7ZcTRPO/view?usp=sharing).
{% endhint %}

### Results

The conversion block will generate a couple of artifacts:&#x20;

* the Hailo-ONNX file,
* and a logs file resembling this one:

```json
[
  {
    "Message": "Converting ONNX to Hailo-ONNX",
    "Data": {
      "Input Path": "/root/model/model.onnx"
    }
  },
  {
    "Message": "Starting conversion",
    "Data": {
      "Target Hardware": "hailo8",
      "Translation": "Done",
      "Reading IO information": "Done",
      "Optimizing Model": "Done",
      "Compiling Model": "Done"
    }
  },
  {
    "Message": "Successful Conversion",
    "Data": {
      "MIME type": "application/x-onnx; device=hailo",
      "Output Path": "/root/model/hailo-model.onnx",
      "Logs Path": "/root/model/hailo-model-logs.json"
    }
  }
]
```
