import sys
import time
import datetime
import traceback
import logging

import torch
from torch import nn
from torch import optim
from torch.nn import functional as func
from torch.utils.data import DataLoader

from parameters import DEVICE, LEARNING_RATE, SCHEDULER_RATE, PATH_TO_MODEL, EPOCHS, BATCH_SIZE, \
    SHUFFLE_DATALOADER, PATH_TO_MODEL_INTER, PATH_TO_LOG, DOWNLOAD_YT_MEDIA, CACHE_DATA, VERBOSE,\
    PATH_TO_TRAINING, PATH_TO_VALIDATION, SPLIT_DATA
import model
import preprocessing
from logger import Logger_custom


def validate(model, dataloader_validation):
    """Calculate loss of model's prediction over validation dataset"""
    val_loss_sum = 0
    for data in dataloader_training:
            time_ref = int(time.time())
            x_audio, x_vision, label = [tensor.to(DEVICE) for tensor in data]
            if VERBOSE: LOGGER.print_log("Validation; time to load data: {}".format(int(time.time() - time_ref)))
            y = model(x_audio, x_vision)
            if VERBOSE: LOGGER.print_log("Validation; time to compute y: {}".format(int(time.time() - time_ref)))            
            val_loss_sum += func.l1_loss(y, [label == i for i in range(2)])
            if VERBOSE: LOGGER.print_log("Validation; time to compute loss: {}".format(int(time.time() - time_ref)))            
    val_loss = val_loss_sum / len(dataloader_validation)
    return val_loss
    

def train(model, epochs, batch, dataloader_training, dataloader_validation):
    """Main training routine"""
    # Init of local var for training
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, SCHEDULER_RATE)
    nb_batch = len(dataloader_training)
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss_sum = 0

        for i, data in enumerate(dataloader_training):

            optimizer.zero_grad()

            x_audio, x_vision = [tensor.to(DEVICE) for tensor in data[:-1]]
            print(data[2])
            label = torch.tensor([data[2] == i for i in range(2)]).to(DEVICE)
            
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

        # Measure and display progress at end of epoch
        epoch_time = datetime.timedelta(seconds=int(time.time() - epoch_start_time))
        epoch_loss = float(epoch_loss_sum / nb_batch)
        validation_loss = validate(model, dataloader_validation)
        LOGGER.print_log("Epoch {}/{} : Time {} Loss {:.3e} Validation loss {:.3e}".format(
            epoch + 1, epochs, epoch_time, epoch_loss, validation_loss), end="\033[K\n")

        # Save and clean up after epoch
        torch.save(model.state_dict(),\
            str(PATH_TO_MODEL_INTER.format(epoch + 1, epochs, str(epoch_loss).replace(".", "p"), SESSION_ID)))
        scheduler.step()
        torch.cuda.empty_cache()

    # Last info display
    total_time = datetime.timedelta(seconds=int(time.time() - start_time))
    validation_loss = validate(model, dataloader_validation)
    LOGGER.print_log("Total Training Time: {} | Last Epoch Loss: {:.3e} | Validation Loss: {:.3e}".format(
        total_time, epoch_loss, validation_loss))

    torch.save(model.state_dict(), str(PATH_TO_MODEL.format(str(epoch_loss).replace(".", "p"), SESSION_ID)))

    return


if __name__ == "__main__":

    #parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    #parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                   help='input batch size for training (default: 64)')
    #parse custom param file

    # Logging init
    SESSION_ID = int(time.time())
    LOGGER = Logger_custom("Global Logger", str(PATH_TO_LOG.format(SESSION_ID)))

    # Data loading
    try:
        if DOWNLOAD_YT_MEDIA and sys.platform == "linux": preprocessing.download_database()
        if CACHE_DATA: preprocessing.cache_data()
        if SPLIT_DATA: preprocessing.split_data()
        
        dataset_training = preprocessing.AVDataset(PATH_TO_TRAINING)
        dataset_validation = preprocessing.AVDataset(PATH_TO_VALIDATION)
        
        dataloader_training = DataLoader(dataset_training, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATALOADER)
        dataloader_validation = DataLoader(dataset_validation, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATALOADER)
    except Exception as e:
        LOGGER.print_log("Error during data loading:\n" + traceback.format_exc(), level=logging.ERROR)
        raise
    
    LOGGER.print_log("\nData was loaded successfully. Training/Validation dataset size: {}/{}".format(
        len(dataloader_training), len(dataloader_validation)))
                     
    # Model init
    model = model.AVENet()
    model = model.float().to(DEVICE)

    LOGGER.print_log("\nModel was loaded successfully. Working device : {}\n".format(DEVICE))
    
    try:
        #validate(model, dataloader_validation)
        train(model, EPOCHS, BATCH_SIZE, dataloader_training, dataloader_validation)
    except Exception as e:
        LOGGER.print_log("Error during training:\n" + traceback.format_exc(), level=logging.ERROR)
        raise
