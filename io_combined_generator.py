import sys
import config
import numpy as np
import os
from os import path

print("\n\n", ("GENERATING COMBINED:START").center(100, "-"), sep="", end="\n\n\n")

audio_src = config.AUDIO_MFCC
video_src = config.VIDEO_NORMALISED
output_src = config.ALIGN_INDEXED
target_src = config.IO_COMBINED

overall_input = []
overall_output = []

max_length = 0

for speaker in os.listdir(audio_src):
    audio_files = os.listdir(path.join(audio_src, speaker))
    video_files = set(os.listdir(path.join(video_src, speaker)))
    output_files = set(os.listdir(path.join(output_src, speaker)))
    random_indexes = np.random.permutation(len(audio_files))

    for audio_file in list(map(lambda i: audio_files[i], random_indexes)):
        if audio_file in video_files and audio_file in output_files:
            audio_npy = np.load(path.join(audio_src, speaker, audio_file))
            video_npy = np.load(path.join(video_src, speaker, audio_file))

            local_max_length = max(audio_npy.shape[0], video_npy.shape[0])
            audio_npy.resize((local_max_length, audio_npy.shape[1]))
            video_npy.resize((local_max_length, video_npy.shape[1]))

            overall_input.append(np.array([ list(i) + list(j) for i, j in zip(audio_npy, video_npy) ]))
            overall_output.append(np.load(path.join(output_src, speaker, audio_file)))
            
            max_length = max(max_length, overall_input[-1].shape[0])

# reshape to make all the arrays of same length
for i in range(len(overall_input)): overall_input[i].resize((max_length, overall_input[i].shape[1]))

# overall data
os.makedirs(path.join(target_src, "overall"), exist_ok=True)
random_indexes = np.random.permutation(len(overall_input))
overall_input = [ overall_input[i] for i in random_indexes ]
overall_output = [ overall_output[i] for i in random_indexes]
np.save(path.join(target_src, "overall", "input"), np.array(overall_input))
np.save(path.join(target_src, "overall", "output"), np.array(overall_output))

# training data
os.makedirs(path.join(target_src, "training"), exist_ok=True)
np.save(path.join(target_src, "training", "input"), np.array(overall_input[ : int(0.80 * len(overall_input))]))
np.save(path.join(target_src, "training", "output"), np.array(overall_output[ : int(0.80 * len(overall_output))]))


# testing data
os.makedirs(path.join(target_src, "testing"), exist_ok=True)
np.save(path.join(target_src, "testing", "input"), np.array(overall_input[int(0.80 * len(overall_input)) : ]))
np.save(path.join(target_src, "testing", "output"), np.array(overall_output[int(0.80 * len(overall_output)) : ]))