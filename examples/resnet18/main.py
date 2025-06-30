import argparse
import typing as tp

import numpy as np
import torch
import torchvision

import npucloud_client


def parse_args():
    parser = argparse.ArgumentParser("Resnet18 NPUCloud example")
    parser.add_argument("--api_token", type=str, required=True,
                        help="Your API token. Get one at https://www.npucloud.tech/payments.php")
    args = parser.parse_args()
    return args


def create_onnx_model(onnx_path: str) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Load a resnet18 model from torchvision and export it to the ONNX format.
    Returns a sample input and a sample output of the model.
    If you're new to ONNX, consider checking out https://netron.app/ to view this file.
    """
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    np.random.seed(4)
    # sample input
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    x_pt = torch.from_numpy(x)
    # Check the output of the original model
    with torch.no_grad():
        out_golden = model(x_pt).numpy()
    # .onnx file is exported with constant shapes. This means that, contrary to the pytorch model,
    # the .onnx file only accepts the inputs of the same shape as x_pt.
    torch.onnx.export(model, (x_pt,), onnx_path, input_names=["input0"], output_names=["output0"])
    return x, out_golden


def main():
    """
    NPUCloud example. Convert resnet18 from torchvision to NPUCloud inference.
    """
    args = parse_args()
    onnx_path = "resnet18.onnx"
    x, out_golden = create_onnx_model(onnx_path)
    print(f"Exported onnx model at {onnx_path}")
    # upload the model to NPUCloud
    model_id = npucloud_client.convert_onnx(onnx_path, args.api_token)
    # run the model with NPUCloud's remote NPU server
    out, profiling_info = npucloud_client.inference(x, model_id, args.api_token)
    # Check the difference between the original PyTorch and NPU's outputs.
    # Some diff is allowed since our NPU device works in fp16 precision.
    # The conversion process handles the output difference and fails if diff exceeds 5%,
    # but we double-check it here.
    diff = np.mean(np.abs(out - out_golden)) / np.mean(np.abs(out_golden))
    assert diff < 0.05, f"Diff={diff} is too large"
    print("Resnet18 model is inferred with NPUCloud successfully")


if __name__ == '__main__':
    main()
