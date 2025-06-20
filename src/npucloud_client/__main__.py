import argparse

import numpy as np

from . import inference


def main():
    """
    Calls the inference from cmd.
    Example:
        python -m npucloud_client --token <YOUR_TOKEN> --model_id <YOUR_MODEL_ID> --inp_shape <YOUR_SHAPE>
    """
    parser = argparse.ArgumentParser(description="Call the inference with random input")
    parser.add_argument('--token', required=True, type=str,
                        help="Inference token. Get your token at https://npucloud.tech/payments.php")
    parser.add_argument('--model_id', required=True, type=str,
                        help="ID of the model to run. Upload your ONNX model at https://npucloud.tech/upload.php "
                             "and get its ID at https://npucloud.tech/models.php")
    parser.add_argument('--array_path', type=str, default="",
                        help="Path to an .npy file with the input data. If not provided, will create "
                             "a random array of shape --inp_shape")
    parser.add_argument('--inp_shape', type=str, default="",
                        help="If provided, create random array of this shape for testing purposes. "
                             "Accept comma-separated string. Example: \"1,3,224,224\"")    
    args = parser.parse_args()
    if args.array_path != "":
        x = np.load(args.array_path)
    else:
        try:
            shape = [int(s) for s in args.inp_shape.split(",")]
        except ValueError as e:
            raise ValueError(f"Could not parse shape {args.shape}") from e
        x = np.random.randn(*shape)
    _, profiling_info = inference(x, args.model_id, args.token)
    print(profiling_info)


if __name__ == '__main__':
    main()
