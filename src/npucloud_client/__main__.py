import argparse

import numpy as np

from . import inference


def test():
    """Call the inference with random input"""
    parser = argparse.ArgumentParser(desc="Call the inference with random input")
    parser.add_argument('--token', required=True, type=str,
                        help="Inference token. Get your token at https://npucloud.tech/payments.php")
    parser.add_argument('--model_id', required=True, type=str,
                        help="ID of the model to run. Upload your ONNX model at https://npucloud.tech/upload.php "
                             "and get its ID at https://npucloud.tech/models.php")
    parser.add_argument('--shape', required=True, type=str,
                        help="Shape of the random input in a comma-separated string format. "
                             "Example: \"1,3,224,224\"")
    parser.add_argument('--n_repeats', type=int,
                        help="N times to repeat the query")
    args = parser.parse_args()
    try:
        shape = [int(s) for s in args.shape.split(",")]
    except ValueError as e:
        raise ValueError(f"Could not parse shape {args.shape}") from e
    x = np.random.randn(*shape)
    for _ in range(args.n_repeats):
        _, profiling_info = inference(x, args.model_id, args.token)
        print(profiling_info)


if __name__ == '__main__':
    test()
