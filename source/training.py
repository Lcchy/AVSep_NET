from torch import nn
from torch import optim

from source.parameters import *
from source.model import *
from source.preprocessing import *
from torch.nn import functional as func


def train(use_model, epochs, batch, use_loader):
    optimizer = optim.Adam(use_model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, SCHEDULER_RATE)

    for epoch in range(epochs):

        for i, data in enumerate(use_loader):

            optimizer.zero_grad()

            print(i, data)
            x_audio, x_vision, labels = data

            y = use_model(x_audio, x_vision)
            loss = func.l1_loss(y, labels)
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

    av_data = AVDataset(PATH_TO_DATASET)
    loader = DataLoader(av_data, batch_size=BATCH_SIZE,
                            shuffle=SHUFFLE_DATALOADER)

    train(model, EPOCHS, BATCH_SIZE, loader)