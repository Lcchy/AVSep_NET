from torch import nn
from torch import optim
from source.parameters import *
from source.model import AVENet


def train(model, epochs, training_data):
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, SCHEDULER_RATE)

    for epoch in range(epochs):

        for i in range(batch):

            optimizer.zero_grad()

            x_audio, x_vision = training_data

            y = model(x_audio, x_vision)
            y = 
