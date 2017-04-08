import sys
import config
import numpy as np
import os
from os import path
from config import config, ALIGN_INDEXED

if len(sys.argv) < 2: 
    print("Need to enter the type of the input")
    exit(1)

print("\n\n", ("GENERATING " + sys.argv.upper() + ":START").center(100, "-"), sep="", end="\n\n\n")

input_src = config[sys.argv[1]]["inputSrc"]
output_src = ALIGN_INDEXED

target_src = config[sys.argv[1]]["datasetPath"]

overall_input = []
overall_output = []

max_length = 0

for speaker in os.listdir(input_src):
    input_files = os.listdir(path.join(input_src, speaker))
    output_files = set(os.listdir(path.join(output_src, speaker)))
    random_indexes = np.random.permutation(len(input_files))

    for input_file in list(map(lambda i: input_files[i], random_indexes)):
        if input_file in output_files:
            overall_input.append(np.load(path.join(input_src, speaker, input_file)))
            overall_output.append(np.load(path.join(output_src, speaker, input_file)))
        
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

print("\n\n", ("GENERATING " + sys.argv.upper() + ":SUCCESS").center(100, "-"), sep="", end="\n\n\n")