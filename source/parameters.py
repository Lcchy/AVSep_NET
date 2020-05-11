import torch
from pathlib import Path

"""MODEL ARCHITECTURE"""
AUDIO_IN_DIM = [257, 200, 1]        #width, height, depth
VISION_IN_DIM = [224, 224, 3]
AVE_NET_EMBED_DIM = 128
CONVNET_CHANNELS = 64
CONVNET_KERNEL = 3
CONVNET_POOL_KERNEL = 2
CONVNET_STRIDE = 2

"""LEARNING PARAMETERS"""
EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 1
SCHEDULER_RATE = 1
SHUFFLE_DATALOADER = False

"""DEVICE PARAMETERS"""
GPU_ON = True
CUDA_ON = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if CUDA_ON and GPU_ON else "cpu")

"""PATHS"""
PATH_TO_MODEL = Path("../model/")
PATH_TO_DATA = Path("../dataset/")
PATH_TO_DATA_CSV = PATH_TO_DATA / "balanced_train_segments.csv"
PATH_TO_VISUAL = PATH_TO_DATA / "media/visual"
PATH_TO_AUDIO = PATH_TO_DATA / "media/audio"

"""EXECUTION PARAMETERS"""
NB_V_FRAMES, NB_A_FRAMES = 1, 100
DOWNLOAD_YT_MEDIA = True    # Only effective if OS is linux
