import sys
import time
import datetime
import traceback
import logging
import gc

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from parameters import DEVICE, LEARNING_RATE, SCHEDULER_RATE, PATH_TO_MODEL, EPOCHS, BATCH_SIZE, \
    SHUFFLE_DATALOADER, PATH_TO_MODEL_INTER, PATH_TO_LOG, DOWNLOAD_YT_MEDIA, PREPARE_DATA, VERBOSE,\
    PATH_TO_TRAINING, PATH_TO_VALIDATION, PATH_TO_DATALIST, WEIGHT_DECAY
import model
import preprocessing
from logger import Logger_custom


def validate(model, dataloader_validation):
    """Calculate loss of model's prediction over validation dataset"""
    with torch.no_grad():
        nb_batch_val = len(dataloader_validation)
        val_loss_sum = 0
        val_accuracy_sum = 0
        past_time = 0
        for (index, data) in enumerate(dataloader_validation):

            if index % 1 + (nb_batch_val // 50) == 0:
                # Progress bar display
                present_time = time.time()
                abs_progress = 100 * (index + 1) / nb_batch_val
                progress = int(abs_progress / 5)
                LOGGER.print_log("Validating model: [{}] {:.2f}% | Time/sample {:.4f}s".format(
                    "=" * progress + " " * (20 - progress),
                    abs_progress,
                    (present_time - past_time) / (BATCH_SIZE * nb_batch_val / 50)),
                    end="\033[K\r")
                past_time = present_time
                
            # Evaluation
            x_audio, x_vision, label, match = [tensor.to(DEVICE) for tensor in data]
            y = model(x_audio, x_vision)
            val_loss_sum += float(F.binary_cross_entropy(y, label.unsqueeze(1)))       # FLOATTTTT
            val_accuracy_sum = float((torch.max(y.float(), dim=2)[1] == match.unsqueeze(1)).sum())

    val_loss = val_loss_sum / BATCH_SIZE
    val_accuracy = val_accuracy_sum / BATCH_SIZE
    return val_loss, val_accuracy


def train(model, dataloader_training, dataloader_validation):
    """Main training routine"""
    # Init of local var for training
    # torch.cuda.empty_cache()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, SCHEDULER_RATE)
    nb_batch = len(dataloader_training)
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        epoch_loss_sum, past_epoch_time = 0, 0

        for i, data in enumerate(dataloader_training):
            optimizer.zero_grad()
            
            x_audio, x_vision, label = [tensor.to(DEVICE) for tensor in data[:-1]]
            
            # Training
            y = model(x_audio, x_vision)
            loss = F.binary_cross_entropy(y, label.unsqueeze(1))
            loss.backward()
            optimizer.step()

            # Progress bar display
            if i % (nb_batch // 50) == 0:
                progress = int(20 * (i + 1) / nb_batch)
                present_epoch_time = time.time() - epoch_start_time
                time_per_sample = (present_epoch_time - past_epoch_time) / (BATCH_SIZE * nb_batch / 50)
                LOGGER.print_log(
                    "Current Epoch {}/{} : [{}]  Time {} Batch {}/{} Loss {:.3f} Time/sample {:.4f}s ETA {}".format(
                    epoch + 1,
                    EPOCHS,
                    "=" * progress + " " * (20 - progress),
                    datetime.timedelta(seconds=int(present_epoch_time)),
                    i + 1,
                    nb_batch,
                    float(loss),
                    time_per_sample,
                    datetime.timedelta(seconds=int(
                        BATCH_SIZE * (nb_batch * (EPOCHS - epoch) - i - 1) * time_per_sample)
                        )
                ), end="\033[K\r")
                past_epoch_time = present_epoch_time
                

            epoch_loss_sum += float(loss)
            break

        # Measure and display progress at end of epoch
        epoch_total_time = datetime.timedelta(seconds=int(time.time() - epoch_start_time))
        epoch_loss = epoch_loss_sum / nb_batch
        val_loss, val_accuracy = validate(model, dataloader_validation)
        LOGGER.print_log("Epoch {}/{} : Time {} Epoch Train Loss {:.3f} Validation Loss {:.3f} Validation Accuracy {:.3f}".format(
            epoch + 1, EPOCHS, epoch_total_time, epoch_loss, val_loss, val_accuracy), end="\033[K\n")

        if epoch % 10 == 0:
            # Save model for later inference or training
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, str(PATH_TO_MODEL_INTER.format(epoch + 1, EPOCHS, SESSION_ID)))
            
        scheduler.step()

    # Last info display
    total_time = datetime.timedelta(seconds=int(time.time() - start_time))
    val_loss, val_accuracy = validate(model, dataloader_validation)
    LOGGER.print_log("Total Training Time {} | Last Epoch Train Loss {:.3f} | Validation Loss {:.3f} | Validation Accuracy {:.3f}".format(
        total_time, epoch_loss, val_loss, val_accuracy))

    torch.save(model.state_dict(), str(PATH_TO_MODEL.format(SESSION_ID)))

    return


if __name__ == "__main__":
    # Todo: parse custom param file

    # Logging init
    SESSION_ID = int(time.time())
    LOGGER = Logger_custom("Global Logger", str(PATH_TO_LOG.format(SESSION_ID)))

    # Data loading
    try:
        if DOWNLOAD_YT_MEDIA and sys.platform == "linux": preprocessing.download_database()
        if PREPARE_DATA:
            train_data_list, validate_data_list = preprocessing.prepare_data()
        else:
            # Load previously saved datalist if not repreparing data
            train_data_list, validate_data_list = torch.load(str(PATH_TO_DATALIST))

        # Load dataset 
        dataset_training = preprocessing.AVDataset(PATH_TO_TRAINING, train_data_list)
        dataset_validation = preprocessing.AVDataset(PATH_TO_VALIDATION, validate_data_list)
        
        dataloader_training = DataLoader(dataset_training, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATALOADER, num_workers=8, drop_last=True)
        dataloader_validation = DataLoader(dataset_validation, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATALOADER, num_workers=8, drop_last=True)
    except Exception as e:
        LOGGER.print_log("Error during data loading:\n" + traceback.format_exc(), level=logging.ERROR)
        raise
    LOGGER.print_log("\nData was loaded successfully. Training/Validation dataset size: {}/{}".format(
        len(train_data_list), len(validate_data_list)))

    # Model init
    model = model.AVENet().float()
    model = model.to(DEVICE)
    LOGGER.print_log("\nModel was loaded successfully. Working device : {}".format(DEVICE), end="\n\n")
    # print(torch.cuda.memory_summary(DEVICE))            

    try:
        train(model, dataloader_training, dataloader_validation)
    except Exception as e:
        LOGGER.print_log("Error during training:\n" + traceback.format_exc(), level=logging.ERROR)
        raise
