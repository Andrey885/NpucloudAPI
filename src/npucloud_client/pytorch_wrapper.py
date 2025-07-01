import numpy as np
import shutil
import tempfile
import typing as tp

import torch

import npucloud_client


class PyTorchWrapperFromModelId(torch.nn.Module):  # inherited from torch.nn.Module to be able to assign as a child module
    """
    Replace a PyTorch module with NPUCloud API call.
    """
    def __init__(self, model_id: str, api_token: str, input_shape: tp.List[int]):
        super().__init__()
        self.model_id = model_id
        self.input_shape = input_shape
        self.api_token = api_token
        self.profiling_info = None

    def get_latest_profiling_info(self) -> tp.Optional[npucloud_client.ProfilingInfo]:
        """Check the profiling info of the last call"""
        return self.profiling_info

    def inference(self, x: np.ndarray) -> np.ndarray:
        """Process the x array with NPUCloud"""
        if self.model_id is None:
            raise ValueError("Please compile NPUCloud PyTorchWrapper by calling PyTorchWrapper.convert(sample_input)")
        out, self.profiling_info = npucloud_client.inference(x, self.model_id, self.api_token)
        return out

    def __call__(self, x):
        """Call the inference"""
        if x.shape != self.input_shape:
            raise ValueError(f"PyTorchWrapper was compiled with shape {self.input_shape}, got shape {x.shape}")
        output = self.inference(x.cpu().detach().numpy())
        # return output of the same dtype and device as input
        output = torch.from_numpy(output).to(x)
        return output


class PyTorchWrapper(PyTorchWrapperFromModelId):
    """
    Convert the PyTorch model into an NPUCloud model.
    This class waits for the first inference and converts to NPUCloud
    statically with the received shape.
    Later on, it expects only the input of this shape and fails otherwise.
    """
    def __init__(self, model: torch.nn.Module, api_token: str, timeout: float = 180,
                 model_name: str = "PyTorchWrapperFromModelId"):
        """
        Converts the model from PyTorch to NPUCloud. The compilation is called at the first inference.
        After that, PyTorchWrapper will expect the same input shape.
        The model expects a single 3D or 4D input with batch size 1.
        Parameters:
            - model: torch.nn.Module
        Model to convert
            - api_token: str
        Your API token from https://www.npucloud.tech/payments.php
            - timeout: float
        Timeout to wait for the conversion process. If exceeded, the script will raise TimeoutError,
        but conversion will proceed in the background. You may view the result at https://www.npucloud.tech/models.php
            - model_name: str
        Your model name at https://www.npucloud.tech/models.php
        """
        super().__init__(None, api_token, None)
        self.model = model
        self.conversion_timeout = timeout
        self.model_name = model_name

    @torch.no_grad()
    def convert(self, x: torch.Tensor):
        """Convert the initially assigned PyTorch model to NPUCloud through ONNX conversion"""
        tmp_dir = tempfile.mkdtemp()
        onnx_path = f"{tmp_dir}/{self.model_name}.onnx"
        try:
            # 1. Export the model to onnx.
            # Using default parameters. Try playing around with opset and other arguments if this line fails.
            torch.onnx.export(self.model, (x,), onnx_path, input_names=["input0"])
            # 2. upload the model to NPUCloud and convert to the NPU format
            self.model_id = npucloud_client.convert_onnx(onnx_path, self.api_token,
                                                         timeout=self.conversion_timeout)
            self.input_shape = x.shape
            self.model = torch.nn.Identity()  # cleanup the model
            shutil.rmtree(tmp_dir)
        except:  # NOQA
            # cleanup before failing
            shutil.rmtree(tmp_dir)
            raise

    def __call__(self, x: torch.Tensor):
        if self.input_shape is None:
            self.convert(x)
        return super().__call__(x)
