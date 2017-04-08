import numpy as np
import scipy.io.wavfile as wav
import random
from pydub import AudioSegment
from python_speech_features import mfcc
from os import path, listdir, makedirs
from config import NOISE_ROOT, AUDIO_RAW, AUDIO_NOISY, AUDIO_NOISY_MFCC
from audio_mfcc_extractor import mfcc_extractor

print("\n\n", ("PROCESSING NOISY AUDIO FILES:START").center(100, "-"), sep="", end="\n\n\n")


noise_audio = AudioSegment.from_file(path.join(NOISE_ROOT, "noise_22kHz.wav"))
for speaker in listdir(AUDIO_RAW):
    for audio_file in listdir(path.join(AUDIO_RAW, speaker)):
        clean_audio = AudioSegment.from_file(path.join(AUDIO_RAW, speaker, audio_file))

        noise_start = random.randint(0, len(noise_audio) - len(clean_audio))
        selected_noise = noise_audio[noise_start: noise_start + len(clean_audio)] - random.randint(7, 12)

        mixed_audio = clean_audio.overlay(selected_noise)

        makedirs(path.join(AUDIO_NOISY, speaker), exist_ok=True)
        mixed_audio.export(path.join(AUDIO_NOISY, speaker, audio_file), format="wav")

mfcc_extractor(AUDIO_NOISY, AUDIO_NOISY_MFCC)


print("\n\n", ("PROCESSING NOISY AUDIO FILES:SUCCESS").center(100, "-"), sep="", end="\n\n\n")