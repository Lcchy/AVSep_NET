import os
import torch
import time
from pathlib import Path


"""EXECUTION PARAMETERS"""
VERBOSE = False
LOG = True
DISPLAY = True
DOWNLOAD_YT_MEDIA = False    # Only effective if OS is linux
PREPARE_DATA = False
SMALL_DATASET = False

"""LEARNING PARAMETERS"""
EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 50
SCHEDULER_RATE = 1
SHUFFLE_DATALOADER = True

"""DEVICE PARAMETERS"""
GPU_ON = True
CUDA_ON = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if CUDA_ON and GPU_ON else "cpu")

"""MODEL ARCHITECTURE"""
AUDIO_IN_DIM = [1, 257, 200]        #width, height, depth
VISION_IN_DIM = [3, 480, 640]
AVE_NET_EMBED_DIM = 128
CONVNET_CHANNELS = 64
CONVNET_KERNEL = 3
CONVNET_POOL_KERNEL = 2
CONVNET_STRIDE = 2

"""MEDIA DONWLOAD FORMAT"""
A_LENGTH = 1
V_LENGTH = 0
A_CODEC = "copy"
A_CHANNELS = 1
A_FREQ = 48000
V_SIZE = (640,480)
V_FRAMERATE = "25/1"
V_CODEC = "copy"
V_ASPECT = "4:3"
V_PIXEL = "+"

"""DATA PROCESSING"""
STFT_N = 512
STFT_HOP = 240
STFT_WINDOW = torch.hann_window(480)
STFT_NORMALIZED = True
SPLIT_RATIO = 0.95

"""PATHS"""
PATH = Path(os.path.dirname(os.path.realpath(__file__))) / ".."
PATH_TO_MODEL = str(PATH / "model/model_loss_{}_id_{}.pt")
PATH_TO_MODEL_INTER = str(PATH / "model/intermediate/model_inter_epoch_{}_of_{}_loss_{}_id_{}.pt")
PATH_TO_DATA = PATH / "dataset"
PATH_TO_MEDIA = PATH_TO_DATA / "media"
PATH_TO_CACHE = PATH_TO_DATA / "cache"
PATH_TO_DATA_CSV = PATH_TO_DATA / "balanced_train_segments.csv"
PATH_TO_VISUAL = PATH_TO_DATA / "media/visual"
PATH_TO_AUDIO = PATH_TO_DATA / "media/audio"
PATH_TO_LOG = str(PATH / "log/log_id_{}")
PATH_TO_TRAINING = PATH_TO_DATA / "training_set"
PATH_TO_VALIDATION = PATH_TO_DATA / "validation_set"
PATH_TO_DATALIST = PATH_TO_DATA / "datalist.pt"

"""LOCAL CHANGES"""
if SMALL_DATASET: 
    PATH_TO_VISUAL = PATH_TO_MEDIA / "visual_small"
    PATH_TO_AUDIO = PATH_TO_MEDIA / "audio_small"

"""TO BE SET BY __MAIN__ IN TRAINING"""
SESSION_ID = None
LOGGER = None