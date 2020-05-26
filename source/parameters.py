import os
import torch
from pathlib import Path

"""MODEL ARCHITECTURE"""
AUDIO_IN_DIM = [1, 257, 200]        #width, height, depth
VISION_IN_DIM = [3, 480, 640]
AVE_NET_EMBED_DIM = 128
CONVNET_CHANNELS = 64
CONVNET_KERNEL = 3
CONVNET_POOL_KERNEL = 2
CONVNET_STRIDE = 2

"""LEARNING PARAMETERS"""
EPOCHS = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 2
SCHEDULER_RATE = 1
SHUFFLE_DATALOADER = True

"""DEVICE PARAMETERS"""
GPU_ON = True
CUDA_ON = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if CUDA_ON and GPU_ON else "cpu")

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
PATH_TO_LOG = PATH / "log"

"""EXECUTION PARAMETERS"""
VERBOSE = True
LOG = True
V_LENGTH, A_LENGTH = 1, 100
DOWNLOAD_YT_MEDIA = False    # Only effective if OS is linux
CACHE_DATA = False
STFT_NORMALIZED = False
SMALL_DATASET = True


"""LOCAL CHANGES"""
if SMALL_DATASET: 
    PATH_TO_VISUAL = PATH_TO_MEDIA / "visual_small"
    PATH_TO_AUDIO = PATH_TO_MEDIA / "audio_small"