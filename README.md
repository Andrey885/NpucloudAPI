<p align="center">
  <img src="https://www.npucloud.tech/img/logo.png" alt="NPUCloud Logo" width="260"/>
</p>

<h1 align="center">NPUCloud Python Client 🚀</h1>

<p align="center">
  <b>Hardware-efficient hosting & fast inference for your AI models.</b><br>
  <a href="https://www.npucloud.tech/"><strong>NPUCloud Official Website »</strong></a>
</p>


<p align="center">

  [![Inference Test](https://github.com/Andrey885/NpucloudAPI/actions/workflows/inference-test.yml/badge.svg)](https://github.com/Andrey885/NpucloudAPI/actions/workflows/inference-test.yml)
  <a href="https://www.npucloud.tech/"><img src="https://img.shields.io/badge/powered%20by-NPUCloud-orange?logo=thunder"></a>
  <a href="https://github.com/Andrey885/NpucloudAPI/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green.svg" alt="License"></a>
</p>

---

## ✨ Why NPUCloud?

- Fast, cost-efficient model inference on real NPUs (Neural Processing Units).
- <b>No hardware or cluster setup needed</b> &mdash; just call from Python, pay only for actual NPU compute!
- Plug & play: immediately use pre-trained models or easily upload your own with open tools.
- Production-ready: secure, robust APIs and built-in profiling for every inference.

---

## 🚀 Quick Start

### 1. Install dependencies
```
python -m pip install numpy requests
```
### 2. Install the NPUCloud client from GitHub
For minimal inference functionality, install with
```
python -m pip install git+https://github.com/Andrey885/NpucloudAPI@main
```
Or, if you want to export models to NpuCloud from ONNX with _npucloud_client_,
```
python -m pip install git+https://github.com/Andrey885/NpucloudAPI@main[onnx]
```
Or, if you want to export models to NpuCloud directly from PyTorch with _npucloud_client_,
```
python -m pip install git+https://github.com/Andrey885/NpucloudAPI@main[all]
```

# 📖 Example
Check out our <a href=https://github.com/Andrey885/NpucloudAPI/tree/main/examples> examples</a> to learn more!

```
import numpy as np
import npucloud_client

# Step 1: Prepare your input (for example, a random image tensor)
x = np.random.randn(1, 3, 224, 224)

# Step 2: Prepare your ONNX model
onnx_path = "your_awesome_model.onnx"

# Step 3: Set your token
token = "YOUR_TOKEN"          # Get your token: https://www.npucloud.tech/payments.php

# Step 4: export your model to NPUCloud (up to 5Gb model size)
model_id = npucloud_client.convert_onnx(onnx_path,token)
# see your model_id at https://www.npucloud.tech/models.php

# Step 5: Run inference on the cloud NPU!
output, profiling_info = npucloud_client.inference(x, model_id, token)

print("Output shape:", output.shape)
print("Profiling info:", profiling_info)
```
---

## 🧩 API Overview
#### convert_onnx(onnx_path: str, token: str) -> str
Convert an .onnx model at onnx_path to NPUCloud's _model_id_. Model size limit is 5Gb.

See how to use this with a PyTorch model at _examples/resnet18_!

- onnx_path (str): path to your .onnx model
- token (str): Your NPUCloud API token (see at [NpuCloud](https://www.npucloud.tech/payments.php)).

Returns:
- model_id (str). The id of your converted model. Use it in the _inference_ function.

#### inference(x: np.ndarray, model_id: str, token: str) -> Tuple[np.ndarray, ProfilingInfo]
Call the model inference with NPUCloud.

- x (np.ndarray): Your input tensor (e.g. image). Will be cast to float16.
- model_id (str): The model catalog ID (see [Your Models](https://www.npucloud.tech/models.php)).
- token (str): Your NPUCloud API token (see at [NpuCloud](https://www.npucloud.tech/payments.php)).

Returns:  
- The output tensor as np.ndarray
- Detailed timing as ProfilingInfo (see below)

#### ProfilingInfo  
_Comprehensive breakdown of time spent per inference:_

- t_task_creation: Task creation time (sec)
- t_input_upload: Upload input time (sec)
- t_compute_queue: Queue wait time (sec)
- npu_compute_time: NPU compute time (sec) <sup><strong>Pay only for this timing!</strong></sup>
- t_result_download: Download result time (sec)
- total_time: End-to-end inference time (sec)

---

## 🛠️ How to Use Your Own Model

- Check out our [Get Started Guide](https://www.npucloud.tech/docs/get_started.php) for model upload, format conversion, and advanced tips!
- Pre-trained models available at [Model Catalog](https://www.npucloud.tech/docs/pretrained_models.php)

---

## 🤝 Requirements

- Python ≥ 3.8
- numpy, requests

---

## 🔗 Useful Links

- 🌐 [NPUCloud Official Website](https://www.npucloud.tech/dashboard.php)
- 📖 [Get an API Token](https://www.npucloud.tech/payments.php)
- 🤖 [Model Catalog](https://www.npucloud.tech/docs/pretrained_models.php)
- 📚 [Docs & Quickstart](https://www.npucloud.tech/docs/get_started.php)

---

## 📜 License

Apache 2.0

---

<p align="center">
Happy inferencing with <b>NPUCloud</b>!<br>
</p>
