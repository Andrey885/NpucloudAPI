import argparse
import logging
import time

import numpy as np
import torch
import torchaudio
import whisper  # install with "$ pip install openai-whisper"

import npucloud_client
from npucloud_client.pytorch_wrapper import PyTorchWrapper, PyTorchWrapperFromModelId

WHISPER_PRECOMPILED_MODEL_ID = "7e60b9f859817f0f8f3d2766e0126688"
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    """Parse args for the whisper demo"""
    parser = argparse.ArgumentParser("Resnet18 NPUCloud example")
    parser.add_argument("--api_token", type=str, required=True,
                        help="Your API token. Get one at https://www.npucloud.tech/payments.php")
    parser.add_argument("--rebuild", action="store_true",
                        help="If passed, will recompile the Whisper model with NPUCloud. It will be exported into "
                             "your account at https://www.npucloud.tech/models.php")
    args = parser.parse_args()
    return args


def _load_sample_audio(whisper_sr: int = 16000):
    sample_wav_path = torchaudio.utils.download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
    waveform, sr = torchaudio.load(sample_wav_path)
    waveform = torchaudio.functional.resample(waveform, sr, whisper_sr)
    if waveform.shape[0] == 2:
        # if the audio is 2-channeled, average the input channels for speech recongition
        waveform = torch.mean(waveform, dim=0, keepdims=True)
    return waveform


def main():
    """Whisper inference example with NPUCloud"""
    t0 = time.perf_counter()
    args = parse_args()
    sample_wav = _load_sample_audio()
    model = whisper.load_model('turbo')
    model.eval()
    LOGGER.info("Whisper model loaded in %.3f sec", time.perf_counter() - t0)
    # convert Whisper encoder from PyTorch to NPUCloud model
    # NOTE: we only convert the audio encoder here.
    # The reasons for that are, firstly, that the audio encoder is the heaviest part of the model,
    # and secondly that NPUCloud currently only supports static inputs shapes. 
    t0 = time.perf_counter()
    if args.rebuild:
        model.encoder = PyTorchWrapper(model.encoder, args.api_token, timeout=600, model_name="whisper_encoder")
    else:
        model.encoder = PyTorchWrapperFromModelId(WHISPER_PRECOMPILED_MODEL_ID,
                                                  args.api_token, (1, 128, 3000))
    LOGGER.info("NpuCloud wrapper created in %.3f sec", time.perf_counter() - t0)
    audio = whisper.pad_or_trim(sample_wav)
    options = whisper.DecodingOptions()
    # make log-Mel spectrogram
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
    result = whisper.decode(model, mel, options)  # first API call to convert the model and heat up NPUCloud's caches
    # decode the audio
    t0 = time.perf_counter()
    result = whisper.decode(model, mel, options)
    LOGGER.info("Inference completed in %.3f sec", time.perf_counter() - t0)
    LOGGER.info("Speech recognition result: %s", result[0].text)
    LOGGER.info("NPU compute time: %.3f", model.encoder.get_latest_profiling_info().npu_compute_time)


if __name__ == '__main__':
    main()
