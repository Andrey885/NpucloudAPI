import time
import base64
import numpy as np
import requests

from .types import CreateTaskResponse, RunTaskResult, ProfilingInfo

API_URL = "https://inference.npucloud.tech"
HEADERS = {"Content-Type": "application/json"}
TIMEOUT = 15


def create_inference_task(model_id: str, token: str, profiling_info: ProfilingInfo
                          ) -> CreateTaskResponse:
    """Notify the backend about the upcoming task and get an AWS url to upload the input"""
    t0 = time.perf_counter()
    # Ask for presigned URL
    payload = {
        "model_id": model_id,
        "token": token
    }
    resp = requests.post(f"{API_URL}/create_task", json=payload, headers=HEADERS, verify=True, timeout=15)
    profiling_info.t_task_creation = round(time.perf_counter() - t0, 3)
    if resp.status_code != 200:
        raise ValueError(f"Got status {resp.status_code} from the server during the inference task creation. "
                         f"Err msg: {resp.text}")
    try:
        resp = CreateTaskResponse(**resp.json())
    except ValueError as e:
        raise ValueError(f"Could not decode create_task's response. Err msg: {resp.text}") from e
    return resp


def upload_input(x: np.ndarray, presigned_url: str, profiling_info: ProfilingInfo) -> None:
    """Upload the model's input to AWS"""
    t0 = time.perf_counter()
    # shape will be restored before inference from the model's io specification
    x_bytes = x.astype(np.float16).reshape(-1).tobytes()
    # Upload file to S3
    resp = requests.put(presigned_url, data=x_bytes, timeout=TIMEOUT)
    resp.raise_for_status()
    profiling_info.t_input_upload = round(time.perf_counter() - t0, 3)


def call_inference(task_id: str, token: str, profiling_info: ProfilingInfo,
                   timeout: float = 60) -> RunTaskResult:
    """Notify npucloud that the input is uploaded, call the model's inference"""
    t0 = time.perf_counter()
    payload = {
        "task_id": task_id,
        "token": token
    }
    resp = requests.post(f"{API_URL}/run_task", json=payload, headers=HEADERS, verify=True, timeout=timeout)
    profiling_info.t_compute_queue = round(time.perf_counter() - t0, 3)
    if resp.status_code != 200:
        raise ValueError(f"Got status {resp.status_code} from the server during the inference call. "
                         f"Err msg: {resp.text}")
    try:
        resp = RunTaskResult(**resp.json())
    except ValueError as e:
        raise ValueError(f"Could not decode create_task's response. Err msg: {resp.text}") from e
    profiling_info.npu_compute_time = resp.npu_compute_time
    return resp


def download_result(inference_data: RunTaskResult, profiling_info: ProfilingInfo) -> np.ndarray:
    """Download the model's output from AWS"""
    t0 = time.perf_counter()
    if len(inference_data.output_encoded) == 0:
        received_data = []
        with requests.get(inference_data.presigned_url, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    received_data.append(chunk)
        received_data = b"".join(received_data)
    else:
        received_data = base64.b64decode(inference_data.output_encoded)
    out = np.frombuffer(received_data, dtype=np.float16)
    out = out.reshape(inference_data.output_shape)
    profiling_info.t_result_download = round(time.perf_counter() - t0, 3)
    return out
