import os
import numpy as np

from python_speech_features import logfbank
import soundfile as sf
import argparse

parser = argparse.ArgumentParser(description='Gender classification.')
parser.add_argument("--flac_path", help="path to the input feature.")
parser.add_argument("--destination_path", help="path to the input feature.")

args = parser.parse_args()


audio_path = str(args.flac_path)
destination_path = str(args.destination_path)

sig, rate = sf.read(os.path.join(audio_path))
fbank_feat = logfbank(sig, rate, nfilt=40, nfft=551)
fbank_feat -= (np.mean(fbank_feat, axis=0) + 1e-8)


np.save(destination_path, fbank_feat)
