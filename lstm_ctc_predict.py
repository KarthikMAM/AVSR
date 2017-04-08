import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
import numpy as np
import util
from os import path
import sys
from config import config, ALIGN_TEXT

config = config[sys.argv[1]]

learningRate, nEpochs, batchSize = 0.001, 1, 64
nFeatures = config["nFeatures"]
nHidden = 128
nClasses = 28 # 26 alphabets, 1 space, 1 blank (for CTC)

logsDir = config["logsDir"]
snapshot = config["snapshot"]

print("\n\n", "LOADING DATA:START".center(100, "-"), sep="", end="\n\n\n")

try:
    inputList = np.load(path.join(config["datasetPath"], "training", "input.npy"), encoding='latin1')
    outputList = np.load(path.join(config["datasetPath"], "training", "output.npy"), encoding='latin1')
except:
    inputList = np.load(path.join(config["datasetPath"], "training", "input.npy"))
    outputList = np.load(path.join(config["datasetPath"], "training", "output.npy"))

# Prediction file details
fileName = "bbaf2n"
speaker = "s1"
expectedOutput = open(path.join(ALIGN_TEXT, speaker, fileName + ".txt")).readlines()[0]

trainBatches, maxLength = util.batchListGen(inputList, outputList, batchSize)
testBatches, _ = util.testBatchGen(fileName + ".npy", speaker, maxLength, type=sys.argv[1], noise=False)
print("SPEAKER         :", ("".join(speaker[1:])).upper())
print("EXPECTED OUTPUT :", expectedOutput.upper())

print("\n\n", "LOADING DATA:SUCCESS".center(100, "-"), sep="", end="\n\n\n")

graph = tf.Graph()
with graph.as_default():
    print("\n\n", "DEFINING GRAPH:START".center(100, "-"), sep="", end="\n\n\n")

    inputX = tf.placeholder(tf.float32, shape=(maxLength, batchSize, nFeatures))
    inputXrs = tf.reshape(inputX, [-1, nFeatures])
    inputList = tf.split(0, maxLength, inputXrs)

    targetIxs = tf.placeholder(tf.int64)
    targetVals = tf.placeholder(tf.int32)
    targetShape = tf.placeholder(tf.int64)
    targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
    
    seqLengths = tf.placeholder(tf.int32, shape=(batchSize))

    weightsOutH1 = tf.Variable(tf.truncated_normal([2, nHidden], stddev=np.sqrt(1 / nHidden)), name=config["weightsOutH1"])
    biasesOutH1 = tf.Variable(tf.zeros([nHidden]), name=config["biasesOutH1"])
    
    weightsOutH2 = tf.Variable(tf.truncated_normal([2, nHidden], stddev=np.sqrt(1 / nHidden)), name=config["weightsOutH2"])
    biasesOutH2 = tf.Variable(tf.zeros([nHidden]), name=config["biasesOutH2"])

    weightsClasses = tf.Variable(tf.truncated_normal([nHidden, nClasses], stddev=np.sqrt(2 / nHidden)), name=config["weightsClasses"])
    biasesClasses = tf.Variable(tf.zeros([nClasses]), name=config["biasesClasses"])

    forwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    backwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)

    fbH1, _, _ = bidirectional_rnn(forwardH1, backwardH1, inputList, dtype=tf.float32, scope=config["fbH1Scope"])

    fbH1rs = [ tf.reshape(t, [batchSize, 2, nHidden]) for t in fbH1 ]
    outH1 = [ tf.reduce_sum(tf.mul(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in fbH1rs ]

    logits = [ tf.matmul(t, weightsClasses) + biasesClasses for t in outH1 ]
    logits3d = tf.pack(logits)

    predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, seqLengths)[0][0])

    for trainVar in tf.trainable_variables(): print("DEFINED: ", trainVar.name)

    print("\n\n", "DEFINING GRAPH:SUCCESS".center(100, "-"), sep="", end="\n\n\n")

print("\n\n", "INITIALIZING:START".center(100, "-"), sep="", end="\n\n\n")
with tf.Session(graph=graph) as session:
    print("\n\n", "INITIALIZING:SUCCESS".center(100, "-"), sep="", end="\n\n\n")
    
    startEpoch = 0

    saver = tf.train.Saver(tf.global_variables())
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir=logsDir)

    if checkpoint:
        try:
            print("\n\n", "LOADING CHECKPOINTS:START".center(100, "-"), sep="", end="\n\n\n")

            startEpoch = int(checkpoint.split("-")[1])

            saver.restore(session, checkpoint)
            for i in tf.trainable_variables():
                session.run(i); print("LOADED: ", i.name)
            
            print("\n\n", "LOADING CHECKPOINTS:SUCCESS".center(100, "-"), sep="", end="\n\n\n")
        except Exception as e:
            print("\n\n", "LOADNIG CHECKPOINTS:ERROR".center(100, "-"), sep="", end="\n\n\n")
    else:
        try:
            session.run([tf.global_variables_initializer()])
        except:
            session.run([tf.initialize_all_variables()])
    
    print("\n\n", "PREDICTION:START".center(100, "-"), sep="", end="\n\n\n")
    batchInput, batchTarget, batchSeqLengths = testBatches[0]
    batchTargetIxs, batchTargetVals, batchTargetShape = batchTarget

    feedDict = {
        inputX: batchInput,
        targetIxs: batchTargetIxs,
        targetShape: batchTargetShape,
        targetVals: batchTargetVals,
        seqLengths: batchSeqLengths
    }

    prediction = session.run([predictions], feed_dict=feedDict)

    mapper = lambda x: " " if x == 0 else chr(ord("a") + x - 1)
    print("PREDICTED STRING :", ("".join(list(map(mapper, prediction[0][1][:prediction[0][2][1]])))).upper().rstrip("P"))
    print("ACTUAL STRING    :", expectedOutput.upper())

    print("\n\n", "PREDICTION:SUCCESS".center(100, "-"), sep="", end="\n\n\n")