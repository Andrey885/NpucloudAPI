from dataclasses import dataclass
import logging
import os
import shutil
import tarfile
import tempfile
import time
import typing as tp

import requests
import onnx

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

API_URL = "https://npucloud.tech"
HEADERS = {"Content-Type": "application/json"}


@dataclass
class CreateUploadTaskResponse:
    model_uuid: str
    s3_presigned_url: str
    file_path: str
    file_name: str


@dataclass
class CheckUploadTaskResponse:
    rknn_model_hash: str
    status: str


def _find_source_files(onnx_path: str) -> tp.List[str]:
    """
    Checks for the dependency files in the .onnx.
    The dependency files can be generated if a large (>2Gb) model is exported.
    """
    model = onnx.load(onnx_path)
    onnx_dir = os.path.split(onnx_path)[0]
    external_files = set([onnx_path])
    for t in model.graph.initializer:
        for entry in t.external_data:
            if entry.key == 'location':
                ext_file_path = f"{onnx_dir}/{entry.value}"
                external_files.add(ext_file_path)
    return list(external_files)


def _tar_source_files(source_files: tp.List[str], tmp_dir: str, onnx_name: str) -> str:
    """
    Pack the list of files into a single .tar file
    """
    tar_path = f"{tmp_dir}/{onnx_name}.tar"
    with tarfile.open(tar_path, "w") as tar:
        for ext_file in source_files:
            tar.add(ext_file, arcname=os.path.split(ext_file)[1])
    return tar_path


def _get_presigned_url(source_file: str, api_key: str) -> CreateUploadTaskResponse:
    """Get S3 presigned_url for the model upload"""
    payload = {"file_name": os.path.split(source_file)[1], "token": api_key}
    resp = requests.post(f"{API_URL}/utils/upload_from_python_client.php", json=payload,
                         headers=HEADERS, verify=True, timeout=15)
    if resp.status_code != 200:
        raise ValueError(f"Got status {resp.status_code} from the server while requesting the model upload. "
                         f"Err msg: {resp.text}")
    try:
        resp = CreateUploadTaskResponse(**resp.json())
    except ValueError as e:
        raise ValueError(f"Could not decode model upload task's response. Err msg: {resp.text}") from e
    return resp


def _upload_file_to_s3(source_file: str, s3_presigned_url: str, timeout: float = 180):
    """Upload the file at source_file to S3 presigned url"""
    with open(source_file, "rb") as f:
        model_bytes = f.read()
    r = requests.put(s3_presigned_url, data=model_bytes, timeout=timeout)
    r.raise_for_status()


def convert_model(source_file: str, api_key: str, timeout: float = 180) -> str:
    """
    Converts the user model (.tar or .onnx file) to NPUCloud's model_id.
    Parameters:
    - source_file: str
        Model source file. Can be an .onnx file, or .tar file if onnx is exported with external dependencies.
    - api_key: str
        Your inference token. Get one at https://npucloud.tech/payments.php
    - timeout: float
        How much time to give for the upload and conversion tasks
    Returns:
    - model_id: str
        Your NPUCloud's model_id. Check your models at https://npucloud.tech/models.php
    """
    # ask for the presigned url
    model_upload_response: CreateUploadTaskResponse = _get_presigned_url(source_file, api_key)
    # upload the model file
    LOGGER.info(f"Uploading model file to NPUCloud. This may take up to {timeout} seconds...")
    _upload_file_to_s3(source_file, model_upload_response.s3_presigned_url, timeout)
    # notify about the upload
    conversion_request = model_upload_response.__dict__
    conversion_request["token"] = api_key
    conversion_request["timeout"] = timeout
    LOGGER.info(f"Waiting for the NPUCloud's conversion process. This may take up to {timeout} seconds...")
    resp = requests.post(f"{API_URL}/utils/call_conversion.php", json=conversion_request,
                         headers=HEADERS, verify=True, timeout=timeout)
    if resp.status_code != 200:
        raise ValueError(f"Got status {resp.status_code} from the server during the model conversion. "
                         f"Err msg: {resp.text}")
    # Wait for the conversion process to complete
    conversion_check_request = {"model_uuid": model_upload_response.model_uuid}
    t0 = time.time()
    while time.time() < t0 + timeout:
        time.sleep(1)
        resp = requests.post(f"{API_URL}/utils/check_conversion.php", json=conversion_check_request,
                             headers=HEADERS, verify=True, timeout=timeout)
        if resp.status_code != 200:
            raise ValueError(f"Got status {resp.status_code} from the server during the model conversion check. "
                             f"Err msg: {resp.text}")
        try:
            resp = CheckUploadTaskResponse(**resp.json())
        except ValueError as e:
            raise ValueError(f"Could not decode model upload check's response. Err msg: {resp.text}") from e
        if resp.rknn_model_hash != "":
            break
    return resp.rknn_model_hash


def convert_onnx(onnx_path: str, api_key: str, timeout: float = 180) -> str:
    """
    Upload the .onnx model at onnx_path to NPUCloud.
    Parameters:
    - onnx_path: str
        Path to the .onnx model file. If exported with external data, the external files are included automatically.
    - api_key: str
        Your inference token. Get one at https://npucloud.tech/payments.php
    - timeout: float
        How much time to give for the upload and conversion tasks
    Returns:
    - model_id: str
        Your NPUCloud's model_id. Check your models at https://npucloud.tech/models.php
    """
    if not onnx_path.endswith(".onnx"):
        raise ValueError(f".onnx extension is expected. Got file path: {onnx_path}")
    source_files = _find_source_files(onnx_path)
    tmp_dir = tempfile.mkdtemp()
    try:
        tar_file = _tar_source_files(source_files, tmp_dir, os.path.split(onnx_path)[1])
        model_id = convert_model(tar_file, api_key, timeout)
    except:  # NOQA
        shutil.rmtree(tmp_dir)
        raise
    shutil.rmtree(tmp_dir)
    LOGGER.info(f"NPUCloud model is converted successfully. Your model_id is {model_id}. "
                "Visit https://npucloud.tech/models.php to learn more!")
    return model_id


def __test():
    """
    Test upload function.
    Usage:
        - get your token at https://npucloud.tech/payments.php
        - python upload_model.py <YOUR_TOKEN>
    """
    import sys
    import torch
    import torchvision
    model = torchvision.models.resnet18()
    model.eval()
    x = torch.zeros((1, 3, 224, 224))
    tmp_dir = tempfile.mkdtemp()
    torch.onnx.export(model, (x,), f"{tmp_dir}/resnet18.onnx")
    api_key = sys.argv[1]
    convert_onnx(f"{tmp_dir}/resnet18.onnx", api_key)
    shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    __test()
