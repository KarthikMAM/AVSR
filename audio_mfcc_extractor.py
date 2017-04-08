import numpy as np
import scipy.io.wavfile as wav
from os import path, listdir, makedirs
from python_speech_features import mfcc
from config import AUDIO_RAW, AUDIO_MFCC 


def get_mfcc(wav_file):
    samplerate, audio = wav.read(wav_file)
    return mfcc(audio, samplerate=samplerate)

def mfcc_extractor(src, dest):
    for speaker in listdir(src):
        makedirs(path.join(dest, speaker), exist_ok=True)
        for audio_file in listdir(path.join(src, speaker)):
            if audio_file.endswith(".wav"):
                mfcc_features = get_mfcc(path.join(src, speaker, audio_file))

                np.save(path.join(dest, speaker, audio_file.split(".")[0]), np.array(mfcc_features))

if __name__ == "__main__":
    print("\n\n", ("PROCESSING CLEAN AUDIO FILES:START").center(100, "-"), sep="", end="\n\n\n")
    mfcc_extractor(AUDIO_RAW, AUDIO_MFCC)
    print("\n\n", ("PROCESSING CLEAN AUDIO FILES:SUCCESS").center(100, "-"), sep="", end="\n\n\n")