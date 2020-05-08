import torch
from pathlib import Path

AUDIO_IN_DIM = [257, 200, 1]        #width, height, depth
VISION_IN_DIM = [224, 224, 3]
AVE_NET_EMBED_DIM = 128
CONVNET_CHANNELS = 64
CONVNET_KERNEL = 3
CONVNET_POOL_KERNEL = 2
CONVNET_STRIDE = 2

EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 1
SCHEDULER_RATE = 1
SHUFFLE_DATALOADER = False

''' DEVICE PARAMETERS '''
GPU_ON = True
CUDA_ON = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if CUDA_ON and GPU_ON else "cpu")


PATH_TO_MODEL = "\model\\"
PATH_TO_DATASET = Path("../dataset/balanced_train_segments_mod2.csv")
