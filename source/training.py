from torch import nn
from torch import optim
from source.parameters import *
from source.model import *
from source.preprocessing import *
from torch.nn import functional as func


def train(use_model, epochs, batch, data):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, SCHEDULER_RATE)

    for epoch in range(epochs):

        for i in range(batch):

            optimizer.zero_grad()

            x_audio, x_vision, labels = data

            y = model(x_audio, x_vision)
            loss = func.l1_loss(y, labels)
            loss.backward()
            optimizer.step()

        torch.save(net.state_dict(), PATH_TO_CHECKPOINT)
        scheduler.step()

        torch.cuda.empty_cache()

    torch.save(net.state_dict(), PATH_TO_MODEL)

    return


if __name__ == "__main__":

    #parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    #parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                   help='input batch size for training (default: 64)')

    model = AVENet()
    training_data = TrainingData()

    Net = DDSPNet().float()
    Net = Net.to(DEVICE)

    dataset = AVDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=SHUFFLE_DATALOADER)

    train(model, EPOCHS, BATCH, dataloader)