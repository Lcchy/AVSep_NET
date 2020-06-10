import matplotlib.pyplot as plt
import os
import shutil
import glob
import time
import torch
import numpy as np

from parameters import PATH_TO_LOG, PATH_TO_VIZ, EPOCHS, PATH_TO_ARCHIVE, PATH


def graph_distrib(tensor_list):
    np_array_list = []
    for tensor in tensor_list:
        np_array_list.append(torch.flatten(tensor).cpu().numpy())
    plt.boxplot(np_array_list, notch=True)
    plt.show()
    plt.close()     


def graph_session(session_id):
    training_loss = []
    validation_loss = []
    mean_epoch_train_loss = []
    val_accuracy = []
    log_file = open(PATH_TO_LOG.format(session_id))
    for line in log_file:
        if "Current Epoch" in line: 
            training_loss.append(float(line.split("Loss ")[1][:5]))
        elif "Validation Loss" in line:
            validation_loss.append(float(line.split("Validation Loss ")[1][:5]))
            mean_epoch_train_loss.append(float(line.split("Epoch Train Loss ")[1][:5]))
            val_accuracy.append(float(line.split("Validation Accuracy ")[1][:5]))

    # Plotting
    fig, ax1 = plt.subplots()
    ax1.plot(training_loss[2:], 'tab:blue')

    ax2 = ax1.twiny()
    ax2.plot(mean_epoch_train_loss, 'sandybrown')
    ax2.plot(validation_loss, 'indianred')
    ax2.plot(val_accuracy, 'mediumseagreen')

    plt.savefig(str(PATH_TO_VIZ / ("fig_session_id_{}.jpg".format(str(session_id)))), dpi=1000)
    plt.close()


def archive_session(session_id):
    assert type(session_id) == str
    session_dir = PATH_TO_ARCHIVE / "session_id_{}".format(session_id)
    try:
        os.mkdir(str(session_dir))
    except FileExistsError:
        pass
    glob_tail = "*{}*".format(session_id)
    file_list = glob.glob(str(PATH / "model" / glob_tail))              \
        + glob.glob(str(PATH / "model" / "intermediate" / glob_tail))   \
        + glob.glob(str(PATH / "log" / glob_tail))                      \
        + glob.glob(str(PATH / "visualisation" / glob_tail))
    for file in file_list:
        os.rename(file, str(session_dir / os.path.basename(file)))


graph_session("1591759898")
# archive_session("1591759898")