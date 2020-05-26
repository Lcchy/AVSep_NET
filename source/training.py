import sys
import time
import datetime

import torch
from torch import nn
from torch import optim
from torch.nn import functional as func
from torch.utils.data import DataLoader

from parameters import DEVICE, LEARNING_RATE, SCHEDULER_RATE, PATH_TO_MODEL, EPOCHS, BATCH_SIZE, \
    SHUFFLE_DATALOADER, PATH_TO_MODEL_INTER, PATH_TO_LOG, DOWNLOAD_YT_MEDIA, CACHE_DATA
from model import AVENet
from preprocessing import AVDataset
from logger import Logger_custom

def train(model, epochs, batch, data_loader):
    # Init of local var for training
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, SCHEDULER_RATE)
    nb_batch = len(data_loader)
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss_sum = 0

        for i, data in enumerate(data_loader):

            optimizer.zero_grad()

            x_audio, x_vision, label = data

            # Training
            y = model(x_audio, x_vision)
            loss = func.l1_loss(y, label)
            loss.backward()
            optimizer.step()

            # Progress bar display
            progress = int(20 * (i + 1) / nb_batch)
            batch_time = datetime.timedelta(seconds=int(time.time() - epoch_start_time))
            LOGGER.print_log("Current Epoch {}/{} : [{}]  Time {} Batch {}/{} Loss {:.3e}".format(
                epoch + 1,
                epochs,
                "=" * progress + " " * (20 - progress),
                batch_time,
                i + 1,
                nb_batch,
                loss
            ), end="\033[K\r")
            epoch_loss_sum += loss

        # Progress display
        epoch_time = datetime.timedelta(seconds=int(time.time() - epoch_start_time))
        epoch_loss = float(epoch_loss_sum / nb_batch)
        LOGGER.print_log("Epoch {}/{} : Time {} Loss {:.3e}".format(epoch + 1, epochs, epoch_time, epoch_loss), end="\033[K\n")

        # Save and clean up after epoch
        torch.save(model.state_dict(),\
            PATH_TO_MODEL_INTER.format(epoch + 1, epochs, str(epoch_loss).replace(".", "p"), SESSION_ID))
        scheduler.step()
        torch.cuda.empty_cache()

    # Last info display
    total_time = datetime.timedelta(seconds=int(time.time() - start_time))
    LOGGER.print_log("Total Training Time: {} | Last Epoch Loss: {:.3e}".format(total_time, epoch_loss))

    torch.save(model.state_dict(), PATH_TO_MODEL.format(str(epoch_loss).replace(".", "p"), SESSION_ID))

    return


if __name__ == "__main__":

    #parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    #parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                   help='input batch size for training (default: 64)')
    #parse custom param file

    # Logging init
    SESSION_ID = int(time.time())
    LOGGER = Logger_custom("Global Logger", PATH_TO_LOG.format(SESSION_ID))

    # Data loading
    av_data = AVDataset()
    if DOWNLOAD_YT_MEDIA and sys.platform == "linux": av_data.download_database()
    if CACHE_DATA: av_data.cache_data()
    loader = DataLoader(av_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATALOADER)

    # Model init
    model = AVENet()
    model = model.float().to(DEVICE)

    LOGGER.print_log("\nWorking device : {}\n".format(DEVICE))

    train(model, EPOCHS, BATCH_SIZE, loader)