import numpy as np
import shutil
import tempfile

import torch

import npucloud_client


class PyTorchWrapper(torch.nn.Module):  # inherited from torch.nn.Module to be able to assign as a child module
    """
    Convert the PyTorch model into an NPUCloud model.
    This class waits for the first inference and converts to NPUCloud
    statically with the received shape.
    Later on, it expects only the input of this shape and fails otherwise.
    """
    def __init__(self, model: torch.nn.Module, api_token: str, timeout: float = 180):
        super().__init__()
        self.model = model
        self.conversion_timeout = timeout
        self.input_shape = None
        self.model_id = None
        self.api_token = api_token

    @torch.no_grad()
    def convert(self, x: torch.Tensor):
        """Convert the initially assigned PyTorch model to NPUCloud through ONNX conversion"""
        tmp_dir = tempfile.mkdtemp()
        onnx_path = f"{tmp_dir}/tmp.onnx"
        try:
            # 1. Export the model to onnx.
            # Using default parameters. Try playing around with opset and other arguments if this line fails.
            torch.onnx.export(self.model, (x,), onnx_path, input_names=["input0"])
            # 2. upload the model to NPUCloud and convert to the NPU format
            self.model_id = npucloud_client.convert_onnx(onnx_path, self.api_token, timeout=self.conversion_timeout)
            self.model = torch.nn.Identity()  # cleanup the model
            shutil.rmtree(tmp_dir)
        except:  # NOQA
            # cleanup before failing
            shutil.rmtree(tmp_dir)
            raise

    def get_latest_profiling_info(self) -> npucloud_client.ProfilingInfo:
        return self.profiling_info

    def inference(self, x: np.ndarray) -> np.ndarray:
        """Process the x array with NPUCloud"""
        if self.model_id is None:
            raise ValueError("Please compile NPUCloud PyTorchWrapper by calling PyTorchWrapper.convert(sample_input)")
        out, self.profiling_info = npucloud_client.inference(x, self.model_id, self.api_key)
        return out

    def __call__(self, x):
        """Call the inference"""
        if x.shape[0] != 1:
            raise ValueError("NPUCloud only supports single-batch inference")
        if self.input_shape is None:
            self.convert(x)
        if x.shape != self.input_shape:
            raise ValueError(f"PyTorchWrapper was compiled with shape {self.input_shape}, got shape {x.shape}")
        output = self.inference(x.cpu().detach().numpy())
        # return output of the same dtype and device as input
        output = torch.from_numpy(output).to(x)
        return output
