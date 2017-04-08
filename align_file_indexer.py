import os
from os import path, listdir, makedirs
import numpy as np
from config import ALIGN_RAW, ALIGN_TEXT, ALIGN_INDEXED

def map_letters(letters):
    return np.array(list(map(
        lambda x: 0 if x == " " else ord(x) - ord("a") + 1,
        list(letters)
    )))

print("\n\n", "INDEXING ALIGN FILES:START".center(100, "-"), sep="", end="\n\n\n")

for speaker in listdir(ALIGN_RAW):
    for align_file in listdir(path.join(ALIGN_RAW, speaker)):
        if align_file.endswith(".align"):
            with open(path.join(ALIGN_RAW, speaker, align_file)) as inp:
                contents = " ".join(list(map(lambda line: line.split()[-1], inp.readlines()[1:-1])))

                makedirs(path.join(ALIGN_TEXT, speaker), exist_ok=True)
                makedirs(path.join(ALIGN_INDEXED, speaker), exist_ok=True)
                
                print(contents, end="", file=open(path.join(ALIGN_TEXT, speaker, align_file.split(".")[0] + ".txt"), "w"))

                np.save(path.join(ALIGN_INDEXED, speaker, align_file.split(".")[0]), map_letters(contents))

print("\n\n", "INDEXING ALIGN FILES:SUCCESS".center(100, "-"), sep="", end="\n\n\n")