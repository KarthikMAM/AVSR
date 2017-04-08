import numpy as np
import config
from os import path


def sparseTensor(arr):
    indices, values = [], []

    for elementId, element in enumerate(arr):
        for valueId, value in enumerate(element):
            indices.append([elementId, valueId])
            values.append(value)
    
    shape = [ len(arr), np.asarray(indices).max(0)[1] + 1 ]

    return (np.array(indices), np.array(values), np.array(shape))

def batchListGen(inputList, outputList, batchSize):
    nFeatures = inputList[0].shape[1]
    maxLength = inputList[0].shape[0]

    randIxs = np.random.permutation(len(inputList))
    start, end = 0, batchSize
    batchList = []

    while end <= len(inputList):
        batchSeqLengths = []
        batchInputs = np.zeros((maxLength, batchSize, nFeatures))
        batchOutputs = []

        for batchId, originalId in enumerate(randIxs[start: end]):
            batchInputs[:, batchId, :] = inputList[originalId]
            batchOutputs.append(outputList[originalId])
            batchSeqLengths.append(inputList[originalId].shape[0])

        batchSeqLengths = np.array(batchSeqLengths)
        batchOutputs = np.array(batchOutputs)
        batchList.append((batchInputs, sparseTensor(batchOutputs), batchSeqLengths))

        start, end = end, end + batchSize
    
    return (batchList, maxLength)

def testBatchGen(fileName, speaker, maxLength, type, noise=False):
    if type == "audio" or type == "audio_noisy":
        npInput = np.load(
            path.join(config.AUDIO_MFCC, speaker, fileName) if not noise 
            else path.join(config.AUDIO_NOISY_MFCC, speaker, fileName)
        )
    elif type == "video":
        npInput = np.load(path.join(config.VIDEO_NORMALISED, speaker, fileName))
    elif type == "combined":
        audioInput = np.load(path.join(config.AUDIO_MFCC, speaker, fileName))
        videoInput = np.load(path.join(config.VIDEO_NORMALISED, speaker, fileName))

        maxLen = max(len(audioInput), len(videoInput))
        audioInput.resize((maxLen, audioInput.shape[1]))
        videoInput.resize((maxLen, videoInput.shape[1]))

        npInput = [ list(i) + list(j) for i, j in zip(audioInput, videoInput) ]
        npInput = np.array(npInput)

    npInput.resize((maxLength, npInput.shape[1]))
    npOutput = np.load(path.join(config.ALIGN_INDEXED, speaker, fileName))

    inputList = np.array([ npInput for i in range(64) ])
    outputList = np.array([ npOutput for i in range(64) ])

    return batchListGen(inputList, outputList, 64)

if __name__ == "__main__":
    print(testBatchGen("bbaf2n.npy", "s1", "Combined", False))

