from torch import nn
from source.parameters import *


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvBlock, self).__init__()

        self.subConvBlock = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, CONVNET_KERNEL, CONVNET_STRIDE),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.convBlock = nn.Sequential(
            self.subConvBlock,
            self.subConvBlock,
            nn.MaxPool2d(CONVNET_POOL_KERNEL)
        )

    def forward(self, x):
        return self.convBlock(x)


class ConvNet(nn.Module):
    def __init__(self, input_channels, first_out_channels=CONVNET_CHANNELS):
        super(ConvNet, self).__init__()

        self.convNet = nn.Sequential(
            ConvBlock(input_channels, first_out_channels),
            ConvBlock(first_out_channels, 2*first_out_channels),
            ConvBlock(2*first_out_channels, 4*first_out_channels),

            nn.Conv2d(4*first_out_channels, 8*first_out_channels, CONVNET_KERNEL, CONVNET_STRIDE),
            nn.BatchNorm2d(8*first_out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(8*first_out_channels, 8*first_out_channels, CONVNET_KERNEL, CONVNET_STRIDE),
            nn.BatchNorm2d(8*first_out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convNet(x)


class AVENet(nn.Module):
    def __init__(self):
        super(AVENet, self).__init__()

        self.visionConv = nn.Sequential(
            ConvNet(VISION_IN_DIM[2]),
            nn.MaxPool2d(14),
            nn.Linear(8*CONVNET_CHANNELS, AVE_NET_EMBED_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(AVE_NET_EMBED_DIM, AVE_NET_EMBED_DIM),
            nn.LayerNorm(AVE_NET_EMBED_DIM)
        )
        self.audioConv = nn.Sequential(
            ConvNet(AUDIO_IN_DIM[2]),
            nn.MaxPool2d((16, 14)),
            nn.Linear(8*CONVNET_CHANNELS, AVE_NET_EMBED_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(AVE_NET_EMBED_DIM, AVE_NET_EMBED_DIM),
            nn.LayerNorm(AVE_NET_EMBED_DIM)
        )

    def forward(self, x_audio, x_vision):
        y_vision, y_audio = self.visionConv(x_vision), self.audioConv(x_audio)
        y = nn.PairwiseDistance(y_vision, y_audio)
        y = nn.Linear(1, 2)
        y = nn.Softmax(2)
        return y


class AVSepNet(nn.Module):
    def __init__(self, audio_input_channels, first_out_channels):
        super(AVSepNet, self).__init__()

        self.visionConv = nn.Sequential(
            ConvNet(VISION_IN_DIM[2]),
        )
        self.audioConv = nn.Sequential(
            ConvNet(AUDIO_IN_DIM[2]),
            nn.MaxPool2d((16, 14)),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )

    def forward(self, x):
        return 1
