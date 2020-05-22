import torch
from torch import nn
from torch import optim
from torch.nn import functional as func
from torch.utils.data import DataLoader

from parameters import DEVICE, LEARNING_RATE, SCHEDULER_RATE, PATH_TO_MODEL, EPOCHS, BATCH_SIZE, SHUFFLE_DATALOADER
from model import AVENet
from preprocessing import AVDataset



def train(use_model, epochs, batch, use_loader):
    optimizer = optim.Adam(use_model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, SCHEDULER_RATE)

    for epoch in range(epochs):

        for i, data in enumerate(use_loader):

            optimizer.zero_grad()

            print(i, data)
            x_audio, x_vision, label = data

            y = use_model(x_audio, x_vision)
            loss = func.l1_loss(y, label)
            loss.backward()
            optimizer.step()

        torch.save(use_model.state_dict(), PATH_TO_MODEL)
        scheduler.step()

        torch.cuda.empty_cache()

    torch.save(use_model.state_dict(), PATH_TO_MODEL)

    return


if __name__ == "__main__":

    #parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    #parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                   help='input batch size for training (default: 64)')

    model = AVENet()

    model = model.float()
    model = model.to(DEVICE)

    av_data = AVDataset(10)
    loader = DataLoader(av_data, batch_size=BATCH_SIZE,
                            shuffle=SHUFFLE_DATALOADER)

    train(model, EPOCHS, BATCH_SIZE, loader)