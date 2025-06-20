from dataclasses import dataclass


@dataclass
class ProfilingInfo:
    """
    Class for storing the npucloud profiling information with the runtimes of each inference step.
    Members:
        - t_task_creation:   time (seconds) to ping npucloud about the upcoming model inference
        - t_input_upload:    time (seconds) to upload the model's input to npucloud's AWS
        - t_compute_queue:   time (seconds) to run the task (including the metadata flow)
        - npu_compute_time:  actual time (seconds) that NPU has spent on the inference. ONLY THIS time will be charged.
        - t_result_download: time (seconds) to upload the model's output from npucloud's AWS
        - total_time:        total time (seconds) of the inference, including the data flows and utility messages
    """
    t_task_creation: float = 0
    t_input_upload: float = 0
    t_compute_queue: float = 0
    npu_compute_time: float = 0
    t_result_download: float = 0
    total_time: float = 0


@dataclass
class RunTaskResult:
    """Dataclass for responces from the inference server"""
    is_status_ok: bool
    npu_compute_time: float
    presigned_url: str
    output_shape: tp.List[int]  # shape of the output np.ndarray
    err_msg: str = ""
    output_encoded: str = ""  # if output is less than 100kb, send it directly through https withput AWS


@dataclass
class CreateTaskResponse:
    """Dataclass for responding to {API_URL}/create_task"""
    task_id: str
    presigned_url: str  # AWS url for the input upload
