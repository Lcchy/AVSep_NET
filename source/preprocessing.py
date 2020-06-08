import sys
import os
import subprocess
import copy
import shutil
import numpy as np
import datetime
import random
import scipy.io.wavfile
from PIL import Image
import torch
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
from parameters import PATH_TO_DATA_CSV, PATH_TO_CACHE, PATH_TO_VISUAL, PATH_TO_AUDIO, PATH_TO_MEDIA, \
    VERBOSE, A_LENGTH, V_LENGTH, A_CODEC, A_CHANNELS, A_FREQ, V_SIZE, V_FRAMERATE, V_CODEC, V_ASPECT, \
    V_PIXEL, STFT_NORMALIZED, SPLIT_RATIO, PATH_TO_TRAINING, PATH_TO_VALIDATION, STFT_N, STFT_HOP,     \
    STFT_WINDOW, STFT_NORMALIZED, PATH_TO_DATALIST
from logger import Logger_custom

LOGGER = Logger_custom("Global Logger")


class AVDataset(Dataset):

    def __init__(self, path, datalist):
        self.data_list = datalist
        self.path = path
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """Returns 3 tensors: audio, visual, match (bool_int)"""
        return self.preprocess_load(index)


### Main functions used to download, process, cache and split the dataset; used in __main__ of training


    def preprocess_load(self, index):
        """Process audio and visual into torch tensor for training."""
        a_filename, v_filename, match = self.data_list[index]

        # Load and preprocess audio
        freq, a_file = scipy.io.wavfile.read(str(self.path / a_filename))
        assert freq == A_FREQ
        a_array = np.copy(a_file)
        a_tensor = torch.tensor(a_array, dtype=torch.float64)
        # Values from the paper are not opt, n_fft > win_length ??!
        spectrogram = torch.stft(a_tensor, STFT_N, STFT_HOP, STFT_WINDOW.size()[0], STFT_WINDOW,
                center=True, pad_mode='reflect', normalized=STFT_NORMALIZED, onesided=True)
        log_spectrogram = torch.log(spectrogram[:,:200,0] ** 2 + 1)     # Format size and set +1 (see noise level) to eliminate log(0)
        # normalize
        a_tensor = torch.unsqueeze(log_spectrogram, 0).float()
        a_tensor = a_tensor.permute(0,2,1)

        # Load and preprocess visual
        v_file = Image.open(str(self.path / v_filename))
        v_file_np = np.copy(v_file)                # OVERHEAD
        v_file.close()
        v_tensor = torch.tensor(v_file_np)
        v_tensor = v_tensor.permute(2,0,1).float()  # Get the torch model input dims right

        label = torch.tensor([float(match==i) for i in range(2)])

        return a_tensor, v_tensor, label, torch.tensor(match)         # To resolve


def prepare_data():
    """Load a correct datalist from the available files, creates matching 
    and non-matching pairs and splits the dataset (by copying) into training
    and validation. 
    Returns the validation and training datalists."""
    data_list = []
    v_file_list = os.listdir(str(PATH_TO_VISUAL))
    datalist_len = len(v_file_list)
    rand_permute = np.random.permutation(len(v_file_list))
    corrupted = 0

    rand_permute = np.random.permutation(datalist_len)
    data_training_id = rand_permute[:int(SPLIT_RATIO * datalist_len)]
    
    shutil.rmtree(str(PATH_TO_TRAINING))
    shutil.rmtree(str(PATH_TO_VALIDATION))
    os.mkdir(str(PATH_TO_TRAINING))
    os.mkdir(str(PATH_TO_VALIDATION))
    train_data_list, validate_data_list = [], []

    for (index, v_filename) in enumerate(v_file_list):
        if index % 100 == 0:
            # Progress bar display
            abs_progress = 100 * (index + 1) / datalist_len
            progress = int(abs_progress / 5)
            LOGGER.print_log("Preparing Data: [{}] {:.2f}%".format("=" * progress + " " * (20 - progress), abs_progress), end="\033[K\r")

        filename = v_filename.split(".")[0]
        match = int(filename[-1])            # Can't use .split("_") because it occurs in filename
        if not match:
            #select random audio to form negative pair by taking the yt_id of a random file
            rand_index = rand_permute[index]
            a_filename = v_file_list[rand_index].split(".")[0][:-2] + "_0.wav"
        else:
            a_filename = filename + ".wav"

        # Check for file sanity
        sane_flag = True
        v_file = Image.open(str(PATH_TO_VISUAL / v_filename))    
        v_file_np = np.copy(v_file)
        v_file.close()
        try:
            a_size = os.stat(str(PATH_TO_AUDIO / a_filename))[6]
        except FileNotFoundError:
            sane_flag = False
            a_size = 0
        if a_size < 96078 or len(v_file_np.shape) < 3: sane_flag = False

        if sane_flag:
            # Insert pair into final datalist and move it to the train/validate folder (random split)
            data_list.append((a_filename, v_filename, match))
            if index in data_training_id:
                dest_path = PATH_TO_TRAINING
                train_data_list.append((a_filename, v_filename, match))
            else:
                dest_path = PATH_TO_VALIDATION
                validate_data_list.append((a_filename, v_filename, match))
            shutil.copy(str(PATH_TO_AUDIO / a_filename), str(dest_path / a_filename))
            shutil.copy(str(PATH_TO_VISUAL / v_filename), str(dest_path / v_filename))

        else:
            corrupted += 1

    # if spectrogram.size()[1] < 200:
    #     pad_size = 200 - spectrogram.size()[1]
    #     spectrogram = F.pad(spectrogram, pad=(0, 0, pad_size // 2, (pad_size + 1) // 2, 0, 0), mode='constant', value=0)
    # Split the chached dataset into training and validation sets.
    # TODO: balanced test split

    torch.save([train_data_list, validate_data_list], str(PATH_TO_DATALIST))
    LOGGER.print_log("Dataset prepared successfully! Skipped {} visual files because data was missing or corrupted".format(corrupted), end="\033[K\n")
    return train_data_list, validate_data_list
        

def download_database():
    """Downoad entire Audio-Visual database by randomly sampling the Youtube videos to the format given in av_parameters.
    Middle sample is for positive label, random for negative."""
    csv_data = pd.read_csv(str(PATH_TO_DATA_CSV, header=2, quotechar='"', skipinitialspace=True))

    # Purge the media directory if necessary
    if len(os.listdir(str(PATH_TO_VISUAL))) + len(os.listdir(str(PATH_TO_AUDIO))) > 0:
        shutil.rmtree(str(PATH_TO_VISUAL))
        shutil.rmtree(str(PATH_TO_AUDIO))
        os.mkdir(str(PATH_TO_VISUAL))
        os.mkdir(str(PATH_TO_AUDIO))
    for index in range(len(csv_data)):
        # Progress bar
        progress = int(20 * (index + 1) / len(csv_data))
        LOGGER.print_log("[" + "=" * progress + " " * (20 - progress) + "]", end="\033[K\r")

        # Actual download calls
        yt_id = csv_data.iloc[index, 0]
        mid_pos = (csv_data.iloc[index, 1] + csv_data.iloc[index, 2]) / 2
        rand_pos = random.uniform(csv_data.iloc[index, 1], csv_data.iloc[index, 2] - 1)    # Taking a 1 sec margin
        yt_download(yt_id, mid_pos, A_LENGTH, V_LENGTH, match=1)
        yt_download(yt_id, rand_pos, A_LENGTH, V_LENGTH, match=0)


### Secondary functions getting called from the above


def yt_download(yt_id, pos, a_length, v_length, match):
    """Download the images and sound of a given youtube video using youtube-dl and ffmpeg by running a bash cmd
    Todo: use a single ffmpeg call with stream mapping."""

    a_cmd_str = form_command(yt_id, pos, a_length, match, leave_out="visual")
    v_cmd_str = form_command(yt_id, pos, v_length, match, leave_out="audio")

    if VERBOSE: LOGGER.print_log("\n---- AUDIO COMMAND: " + a_cmd_str + "\n")

    subprocess.run(a_cmd_str, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # shell=True is a security issue

    if VERBOSE: LOGGER.print_log("\n---- VISUAL COMMAND: " + v_cmd_str + "\n")

    subprocess.run(v_cmd_str, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def form_command(yt_id, pos, length, match, leave_out):
    """Parse arguments into command to form the bash execution string"""

    cmd = {
        'stream handler': "ffmpeg",
        'input flags': {
            'overwrite': "-y",
            'position': "-ss {}".format(str(datetime.timedelta(seconds=pos))),                            # format: 00:00:00
            'duration': "-t {}".format(str(datetime.timedelta(seconds=length))) if length > 0 else "",                  #
            'input': "-i",
        },
        'youtube-dl': {
            'head': "$(",
            'main': "/usr/local/bin/youtube-dl",
            'quality': "-f best",
            'url flag': "--get-url",
            'url': "https://www.youtube.com/watch?v=" + yt_id,
            'tail': ")",
        },
        'output': {
            'visual': {
                'nb visual frames': "-frames:v 1" if length == 0 else "",
                'framerate': "-r {}".format(V_FRAMERATE) if length > 0 else "",
                'image size': "-s {}x{}".format(V_SIZE[0], V_SIZE[1]),     #INPUT_DIM
                # 'birate': '-b {}',
                #'pixel format': "-pix_fmt {}".format(V_PIXEL),
                #'video codec': "-codec:v {}".format(V_CODEC),
                #'aspect': "-aspect {}".format(V_ASPECT),
                'output path': str(PATH_TO_VISUAL / ("{}_{}".format(yt_id, match) + ".jpg" if length == 0 else ".mp4")),
            },
            'audio': {
                #'audio codec': "-codec:a {}".format(A_CODEC),
                #'layout':
                # 'birate': '-b {}',
                'frequency': "-ar {}".format(A_FREQ),
                'channels': "-ac {}".format(A_CHANNELS),
                'output path': str(PATH_TO_AUDIO / ("{}_{}".format(yt_id, match) + ".wav")),
            },
        },
    }

    cmd_str = custom_join(cmd, leave_out=[leave_out])
    return cmd_str


def custom_join(d, out="", leave_out=[]):
    """Join a dict of strings recursively using an arg accumulator.
    Leaves out part of dict if key is in given list.
    Terminates because I believe."""
    if type(d) == dict and len(d) == 0:
        return ""
    elif type(d) != dict:
        return (str(d) + " " + out).rstrip()
    else:
        first_key = list(d)[0]
        head = d[first_key] if first_key not in leave_out else {}
        tail = {key:value for (key,value) in d.items() if key != first_key and not key in leave_out}
        return (custom_join(head, out, leave_out) + " " + custom_join(tail, out, leave_out)).rstrip()


def install_yt_dl():
    """Download and install youtube-dl and ffmpeg for latest working version.
    To run manually."""
    assert sys.platform == "linux"
    cmd1 = "sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl"
    cmd2 = "sudo chmod a+rx /usr/local/bin/youtube-dl"
    cmd3 = "yes | sudo apt-get install ffmpeg"
    subprocess.run(cmd1.split(" "), shell=True)
    subprocess.run(cmd2.split(" "), shell=True)
    subprocess.run(cmd3.split(" "), shell=True)