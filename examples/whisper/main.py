import argparse
import shutil
import time
import tempfile

import numpy as np
import torch
import torchaudio
import whisper  # install with pip install openai-whisper

import npucloud_client
from npucloud_client.pytorch_wrapper import PyTorchWrapper


def parse_args():
    parser = argparse.ArgumentParser("Resnet18 NPUCloud example")
    parser.add_argument("--api_token", type=str, required=True,
                        help="Your API token. Get one at https://www.npucloud.tech/payments.php")
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
    t0 = time.perf_counter()
    args = parse_args()
    sample_wav = _load_sample_audio()
    model = whisper.load_model('turbo')
    model.eval()
    print("model loaded", time.perf_counter() - t0)
    t0 = time.perf_counter()
    # convert Whisper encoder from PyTorch to NPUCloud model
    model.encoder = PyTorchWrapper(model.encoder, args.api_token)
    print("rknn created", time.perf_counter() - t0)
    t0 = time.perf_counter()
    audio = whisper.pad_or_trim(sample_wav)
    # make log-Mel spectrogram
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
    # decode the audio
    print("mel", mel.shape)
    t0 = time.perf_counter()
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    print("text inferred", time.perf_counter() - t0)
    print(result)
    exit()


if __name__ == '__main__':
    main()
