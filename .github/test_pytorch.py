import sys
import tempfile
import shutil

import numpy as np
import torch

import npucloud_client


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = torch.mean(x, dim=(1, 2, 3))
        return x


def create_onnx_model(onnx_path: str):
    model = SimpleModel()
    model.eval()
    model = model.half().float()
    x = np.random.randn(1, 3, 16, 16).astype(np.float16).astype(np.float32)
    x_pt = torch.from_numpy(x)
    with torch.no_grad():
        out_golden = model(x_pt).numpy()
    torch.onnx.export(model, (x_pt,), onnx_path, input_names=["input0"])
    return x, out_golden


def test():
    """
    Test upload function. Uploads SimpleModel to NPUCloud, infers with pytorch and npucloud,
    checks the output difference.
    Usage:
        - get your token at https://www.npucloud.tech/payments.php
        - python test.py <YOUR_TOKEN>
    """
    tmp_dir = tempfile.mkdtemp()
    onnx_path = f"{tmp_dir}/SimpleModel.onnx"
    # create a model
    np.random.seed(4)
    x, out_golden = create_onnx_model(onnx_path)
    api_key = sys.argv[1]
    # upload the model to NPUCloud
    model_id = npucloud_client.convert_onnx(f"{tmp_dir}/SimpleModel.onnx", api_key)
    shutil.rmtree(tmp_dir)
    # run the model
    out, profiling_info = npucloud_client.inference(x, model_id, api_key)
    # check output
    diff = np.mean(np.abs(out - out_golden)) / np.mean(np.abs(out_golden))
    print(f"diff={diff:.5f}, shape={out.shape}, {profiling_info}")
    # 5% diff is allowed. It's related to fp16 rknn conversion
    assert diff < 0.05, f"Diff={diff} is too large"
    # delete the model
    npucloud_client.delete_model(api_key, model_id)


if __name__ == '__main__':
    test()
