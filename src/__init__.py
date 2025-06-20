import time
import typing as tp

import numpy as np

from . import core
from .types import ProfilingInfo, CreateTaskResponse, RunTaskResult


def inference(x: np.typing.NDArray, model_id: str, token: str
              ) -> tp.Tuple[np.typing.NDArray[np.float16], ProfilingInfo]:
    """
    Call the npucloud inference.
    Parameters:
        x : np.ndarray
            Model's input. This data will be cast to float16 and put to the model you've uploaded as is
            (no norm or other preprocessing is applied)
        model_id: str
            ID of the model to run. Upload your ONNX model at https://npucloud.tech/upload.php
            and get its ID at https://npucloud.tech/models.php
        token: str
            Inference token. Get your token at https://npucloud.tech/payments.php
    Returns:
        np.typing.NDArray[np.float16]
            Output array. Matches the output of the .onnx model uploaded by user with no more than
            5% relative MAE on average. No postprocessing is applied except for casting to float16
        ProfilingInfo
            Timing information about each of the inference steps. This includes creating the inference task,
            uploading the input by client, calling the inference and downloading the output
    """
    t0 = time.perf_counter()
    profiling_info = ProfilingInfo()
    created_task: CreateTaskResponse = core.create_inference_task(model_id, token, profiling_info)
    # upload the input to AWS
    core.upload_input(x, created_task.presigned_url, profiling_info)
    # call npucloud's model inference
    inference_data: RunTaskResult = core.call_inference(created_task.task_id, token, profiling_info)
    # download the inference result
    output = core.download_result(inference_data, profiling_info)
    profiling_info.total_time = round(time.perf_counter() - t0, 3)
    return output, profiling_info
