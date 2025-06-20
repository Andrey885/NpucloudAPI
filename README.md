# NpucloudAPI
Hardware-efficient hosting for your AI models.

This is an official Python client for model inference with [NPUCloud](https://npucloud.tech/).


## Quick Start

### Install dependencies:

```
python -m pip install numpy requests
```
### Install NpucloudAPI:
```
python -m pip install git+https://github.com/Andrey885/NpucloudAPI@main
```

### Example usage:

```
import numpy as np
from npucloud_client import inference

# Example input data; replace with yours:
x = np.random.randn(1, 3, 224, 224)

# Your credentials:
model_id = "YOUR_MODEL_ID"    # see https://npucloud.tech/models.php
token = "YOUR_TOKEN"          # get your token at https://npucloud.tech/payments.php

# Run inference
output, profiling_info = inference(x, model_id, token)
print("Output shape:", output.shape)
print("Profiling info:", profiling_info)
```
---

## API

#### inference(x: np.ndarray, model_id: str, token: str) -> Tuple[np.typing.NDArray[np.float16], ProfilingInfo]

- **x** (np.ndarray): Your input tensor (e.g., an image). Will be cast to float16.
- model_id (str): The ID of your model on NPUCloud.
- token (str): Your account token.

Returns:
- The output ndarray,
- a profiling structure (ProfilingInfo) containing timing details for each inference step.

#### ProfilingInfo
A dataclass with the following fields:
- t_task_creation: Time spent for task creation (sec)
- t_input_upload: Time spent uploading input (sec)
- t_compute_queue: Time spent in the compute queue (sec)
- npu_compute_time: Actual time spent on inference on NPU (sec)
- t_result_download: Time spent downloading the result (sec)
- total_time: Total end-to-end time (sec)

### Other docs
See https://npucloud.tech/docs/get_started.php for more info about preparing your model and examples with the existing models.

---

## Requirements

- Python >= 3.8
- numpy
- requests

---

## Useful Links

- [NPUCloud — Official Website](https://npucloud.tech/)
- [Get your token](https://npucloud.tech/payments.php)
- [Your model catalog](https://npucloud.tech/models.php)

---

## License

Apache 2.0
