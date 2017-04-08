from os import path

ROOT = "/mnt/C680613280612A5F/AVSR"
DATA_ROOT = path.join(ROOT, "data")
MISC_ROOT = path.join(ROOT, "misc")

CLASSIFIER_ROOT = path.join(MISC_ROOT, "classifier")
NOISE_ROOT = path.join(MISC_ROOT, "noise")

ALIGN_ROOT = path.join(DATA_ROOT, "align")
ALIGN_RAW = path.join(ALIGN_ROOT, "raw")
ALIGN_TEXT = path.join(ALIGN_ROOT, "txt")
ALIGN_INDEXED = path.join(ALIGN_ROOT, "indexed")

VIDEO_ROOT = path.join(DATA_ROOT, "video")
VIDEO_RAW = path.join(VIDEO_ROOT, "raw")
VIDEO_FRAMES = path.join(VIDEO_ROOT, "frames")
VIDEO_EXTRACT = path.join(VIDEO_ROOT, "extract")
VIDEO_NORMALISED = path.join(VIDEO_ROOT, "normalised")

AUDIO_ROOT = path.join(DATA_ROOT, "audio")
AUDIO_RAW = path.join(AUDIO_ROOT, "raw")
AUDIO_MFCC = path.join(AUDIO_ROOT, "mfcc")
AUDIO_NOISY = path.join(AUDIO_ROOT, "noisy")
AUDIO_NOISY_MFCC = path.join(AUDIO_ROOT, "noisy_mfcc")

IO_ROOT = path.join(DATA_ROOT, "io")
IO_AUDIO = path.join(IO_ROOT, "audio")
IO_VIDEO = path.join(IO_ROOT, "video")
IO_COMBINED = path.join(IO_ROOT, "combined")
IO_AUDIO_NOISY = path.join(IO_ROOT, "audio_noisy")

LOG_ROOT = path.join(ROOT, "log")
LOGS_AUDIO = path.join(LOG_ROOT, "audio")
LOGS_VIDEO = path.join(LOG_ROOT, "video")
LOGS_COMBINED = path.join(LOG_ROOT, "combined")

config = {
    "audio": {
        "logsDir": "logs/audio",
        "snapshot": "CTC_AUDIO",
        "nFeatures": 13,
        "datasetPath": IO_AUDIO,
        "inputSrc": AUDIO_MFCC,
        "weightsOutH1": "Variable",
        "biasesOutH1": "Variable_1",
        "weightsOutH2": "Variable_2",
        "biasesOutH2": "Variable_3",
        "weightsClasses": "Variable_4",
        "biasesClasses": "Variable_5",
        "fbH1Scope": "BDLSTM_H1"
    },
    "audio_noisy": {
        "logsDir": "logs/audio",
        "snapshot": "CTC_AUDIO",
        "nFeatures": 13,
        "datasetPath": IO_AUDIO_NOISY,
        "inputSrc": AUDIO_NOISY_MFCC,
        "weightsOutH1": "Variable",
        "biasesOutH1": "Variable_1",
        "weightsOutH2": "Variable_2",
        "biasesOutH2": "Variable_3",
        "weightsClasses": "Variable_4",
        "biasesClasses": "Variable_5",
        "fbH1Scope": "BDLSTM_H1"
    },
    "video": {
        "logsDir": "logs/video",
        "snapshot": "CTC_VIDEO",
        "nFeatures": 2,
        "datasetPath": IO_VIDEO,
        "inputSrc": VIDEO_NORMALISED,
        "weightsOutH1": "Variable",
        "biasesOutH1": "Variable_1",
        "weightsOutH2": "Variable_2",
        "biasesOutH2": "Variable_3",
        "weightsClasses": "Variable_4",
        "biasesClasses": "Variable_5",
        "fbH1Scope": "BDLSTM_H1_Video"
    },
    "combined": {
        "logsDir": "logs/combined",
        "snapshot": "CTC_COMBINED",
        "nFeatures": 15,
        "datasetPath": IO_COMBINED,
        "weightsOutH1": "weightsOutH1Combined",
        "biasesOutH1": "biasesOutH1Combined",
        "weightsOutH2": "weightsOutH2Combined",
        "biasesOutH2": "biasesOutH2Combined",
        "weightsClasses": "weightsClassesCombined",
        "biasesClasses": "biasesClassesCombined",
        "fbH1Scope": "BDLSTM_H1_Combined"
    }
}