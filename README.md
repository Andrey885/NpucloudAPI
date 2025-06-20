<p align="center">
  <img src="https://npucloud.tech/img/logo.png" alt="NPUCloud Logo" width="260"/>
</p>

<h1 align="center">NPUCloud Python Client 🚀</h1>

<p align="center">
  <b>Hardware-efficient hosting & ultra-fast inference for your AI models.</b><br>
  <a href="https://npucloud.tech/"><strong>NPUCloud Official Website »</strong></a>
</p>

<p align="center">
  <a href="https://github.com/Andrey885/NpucloudAPI/actions"><img src="https://img.shields.io/github/actions/workflow/status/Andrey885/NpucloudAPI/ci.yml?branch=main" alt="CI"></a>
  <a href="https://python.org/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python3"></a>
  <a href="https://npucloud.tech/"><img src="https://img.shields.io/badge/powered%20by-NPUCloud-orange?logo=thunder"></a>
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
```
python -m pip install git+https://github.com/Andrey885/NpucloudAPI@main
```
### 3. Inference Example
```
import numpy as np
from npucloud_client import inference

# Step 1: Prepare your input (for example, a random image tensor)
x = np.random.randn(1, 3, 224, 224)

# Step 2: Set your model & auth details
model_id = "YOUR_MODEL_ID"    # See: https://npucloud.tech/models.php
token = "YOUR_TOKEN"          # Get your token: https://npucloud.tech/payments.php

# Step 3: Run inference on the cloud NPU!
output, profiling_info = inference(x, model_id, token)

print("Output shape:", output.shape)
print("Profiling info:", profiling_info)
```
---

## 🧩 API Overview

#### inference(x: np.ndarray, model_id: str, token: str) -> Tuple[np.ndarray, ProfilingInfo]

- x (np.ndarray): Your input tensor (e.g. image). Will be cast to float16.
- model_id (str): The model catalog ID (see [Your Models](https://npucloud.tech/models.php)).
- token (str): Your NPUCloud API token (see at [NpuCloud](https://npucloud.tech/payments.php)).

Returns:  
- The output tensor as np.ndarray
- Detailed timing as ProfilingInfo (see below)

---

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

- Check out our [Get Started Guide](https://npucloud.tech/docs/get_started.php) for model upload, format conversion, and advanced tips!
- Pre-trained models available at [Model Catalog](https://npucloud.tech/docs/pretrained_models.php)

---

## 🤝 Requirements

- Python ≥ 3.8
- numpy, requests

---

## 🔗 Useful Links

- 🌐 [NPUCloud Official Website](https://npucloud.tech/dashboard.php)
- 📖 [Get an API Token](https://npucloud.tech/payments.php)
- 🤖 [Model Catalog](https://npucloud.tech/docs/pretrained_models.php)
- 📚 [Docs & Quickstart](https://npucloud.tech/docs/get_started.php)

---

## 📜 License

Apache 2.0

---

<p align="center">
Happy inferencing with <b>NPUCloud</b>!<br>
<i>Performance & Simplicity.</i>
</p>
