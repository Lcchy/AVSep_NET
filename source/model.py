from torch import nn
import torch
import sys
from parameters import CONVNET_KERNEL, CONVNET_STRIDE, CONVNET_POOL_KERNEL, CONVNET_CHANNELS,\
    CONVNET_CHANNELS, AVE_NET_EMBED_DIM, VISION_IN_DIM, AUDIO_IN_DIM, BATCH_SIZE, DEVICE


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        self.convBlock = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, CONVNET_KERNEL, stride, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(output_channels, output_channels, CONVNET_KERNEL, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(CONVNET_POOL_KERNEL)
        )

    def forward(self, x):
        return self.convBlock(x)


class ConvNet(nn.Module):
    def __init__(self, input_channels, first_out_channels=CONVNET_CHANNELS):
        super().__init__()
        self.convNet = nn.Sequential(
            ConvBlock(input_channels, first_out_channels, CONVNET_STRIDE),
            ConvBlock(first_out_channels, 2*first_out_channels),
            ConvBlock(2*first_out_channels, 4*first_out_channels),

            nn.Conv2d(4*first_out_channels, 8*first_out_channels, CONVNET_KERNEL, padding=1),
            nn.BatchNorm2d(8*first_out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(8*first_out_channels, 8*first_out_channels, CONVNET_KERNEL, padding=1),
            nn.BatchNorm2d(8*first_out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convNet(x)


class AVENet(nn.Module):
    def __init__(self):
        super().__init__()

        self.visionConv = nn.Sequential(
            ConvNet(input_channels=VISION_IN_DIM[0]),
            nn.MaxPool2d(kernel_size=(VISION_IN_DIM[1] // 16,VISION_IN_DIM[2] // 16))
        )
        self.audioConv = nn.Sequential(
            ConvNet(input_channels=AUDIO_IN_DIM[0]),
            nn.MaxPool2d(kernel_size=(AUDIO_IN_DIM[2] // 16, AUDIO_IN_DIM[1] // 16))
        )
        self.linearNorm = nn.Sequential(
            nn.Linear(in_features=8*CONVNET_CHANNELS, out_features=AVE_NET_EMBED_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=AVE_NET_EMBED_DIM, out_features=AVE_NET_EMBED_DIM),
            nn.LayerNorm(normalized_shape=AVE_NET_EMBED_DIM, elementwise_affine=False)
        )
        self.topLinear = nn.Sequential(
            nn.Linear(in_features=1, out_features=2),
            nn.Softmax(dim=2)                           # yes
        )

    def forward(self, x_audio, x_vision):
        y_vision = self.visionConv(x_vision)
        y_vision = y_vision.permute(0,2,3,1)    # Put the channels as last for the linear layer to take them as features
        y_vision = self.linearNorm(y_vision)

        y_audio = self.audioConv(x_audio)
        y_audio = y_audio.permute(0,2,3,1)
        y_audio = self.linearNorm(y_audio)
        y = torch.zeros((BATCH_SIZE, 1)).to(DEVICE)
        for i in range(BATCH_SIZE):
            y[i] = torch.dist(y_audio[i], y_vision[i])      #oh god : torch.nn.PairwiseDistance
        return self.topLinear(y.unsqueeze(1))


class AVSepNet(nn.Module):
    def __init__(self, audio_input_channels, first_out_channels):
        super().__init__()

        self.visionConv = nn.Sequential(
            ConvNet(input_channels=VISION_IN_DIM[2]),
        )
        self.audioConv = nn.Sequential(
            ConvNet(input_channels=AUDIO_IN_DIM[2]),
            nn.MaxPool2d(kernel_size=(16, 12)),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=128),
        )

    def forward(self, x):
        return 1    # TODO
