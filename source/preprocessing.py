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
import sys
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
from parameters import PATH_TO_DATA_CSV, PATH_TO_CACHE, PATH_TO_VISUAL, PATH_TO_AUDIO, PATH_TO_MEDIA, \
    VERBOSE, A_LENGTH, V_LENGTH, A_CODEC, A_CHANNELS, A_FREQ, V_SIZE, V_FRAMERATE, V_CODEC, V_ASPECT, \
    V_PIXEL, STFT_NORMALIZED, SPLIT_RATIO, PATH_TO_TRAINING, PATH_TO_VALIDATION, STFT_N, STFT_HOP,     \
    STFT_WINDOW, STFT_NORMALIZED
from logger import Logger_custom

LOGGER = Logger_custom("Global Logger")


class AVDataset(Dataset):

    def __init__(self, path):
        self.path = path
        self.data_list = os.listdir(self.path)
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """Returns 3 tensors: audio, visual, match (bool_int)"""
        return torch.load(self.path / self.data_list[index])


### Main functions used to download, process, cache and split the dataset; used in __main__ of training

def download_database():
    """Downoad entire Audio-Visual database by randomly sampling the Youtube videos to the format given in av_parameters.
    Middle sample is for positive label, random for negative."""
    csv_data = pd.read_csv(PATH_TO_DATA_CSV, header=2, quotechar='"', skipinitialspace=True)

    # Purge the media directory if necessary
    if len(os.listdir(PATH_TO_VISUAL)) + len(os.listdir(PATH_TO_AUDIO)) > 0:
        shutil.rmtree(PATH_TO_VISUAL)
        shutil.rmtree(PATH_TO_AUDIO)
        os.mkdir(PATH_TO_VISUAL)
        os.mkdir(PATH_TO_AUDIO)
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


def cache_data():
    """Process and store the downloaded media into torch tensor format in cache folder.
    Pair up visual and audio parts randomly to create false pairs.
    Assumes that the file lists in visual and audio correspond."""
    v_file_list = os.listdir(PATH_TO_VISUAL)
    rand_permute = np.random.permutation(len(v_file_list))

    for (index, v_filename) in enumerate(v_file_list):
        # Progress bar
        float_progress = (index + 1) / len(v_file_list)
        progress = int(20 * (index + 1) / len(v_file_list))
        LOGGER.print_log("Caching the data: [{}] {:.2f}%".format(
            "=" * progress + " " * (20 - progress), 100 * float_progress), end="\033[K\r")

        filename = v_filename.split(".")[0]
        match = int(filename[-1])            # Can't use .split("_") because it occurs in filename

        v_file = Image.open(PATH_TO_VISUAL / v_filename)
        v_file = np.copy(v_file)
        v_tensor = torch.tensor(v_file)
        v_tensor = v_tensor.permute(2,0,1).float()  # Get the torch model input dims right

        if not match:
            #select random audio to form negative pair by taking the yt_id of a random file
            rand_index = rand_permute[index]
            a_filename = v_file_list[rand_index].split(".")[0][:-2] + "_0.wav"
        else:
            a_filename = filename + ".wav"

        freq, a_file = scipy.io.wavfile.read(PATH_TO_AUDIO / a_filename)
        assert freq == A_FREQ
        a_tensor = process_audio(a_file)
        a_tensor = torch.unsqueeze(a_tensor, 0).float()
        a_tensor = a_tensor.permute(0,2,1)

        match = torch.tensor(match)

        if VERBOSE: LOGGER.print_log("Caching {} and {} into {} with dims: {} | {} | {}".format(
            a_filename, v_filename, filename, a_tensor.size(), v_tensor.size(), match.size()
        ), end="\033[K\n")
        torch.save((a_tensor, v_tensor, match), PATH_TO_CACHE / (filename + ".pt"))
    LOGGER.print_log("Done caching!", end="\033[K\n")
        

def split_data():
    """Split the chached dataset into training and validation sets.
    TODO: balanced test split"""
    data_list = os.listdir(PATH_TO_CACHE)
    data_nb = len(data_list)
    rand_permute = np.random.permutation(data_nb)
    data_training_id = rand_permute[:int(SPLIT_RATIO * data_nb)]
     
    shutil.rmtree(PATH_TO_TRAINING)
    shutil.rmtree(PATH_TO_VALIDATION)
    os.mkdir(PATH_TO_TRAINING)
    os.mkdir(PATH_TO_VALIDATION)
    
    for (id, file) in enumerate(data_list):
        if id in data_training_id:
            shutil.copy(PATH_TO_CACHE / file, PATH_TO_TRAINING / file)
        else:
            shutil.copy(PATH_TO_CACHE / file, PATH_TO_VALIDATION / file)

### Secondary functions getting called from the above


def process_audio(a_file):
    """Process audio before caching. STFT"""
    a_array = np.copy(a_file)
    a_tensor = torch.tensor(a_array, dtype=torch.float64)
    print(max(a_tensor))
    # Values from the paper are not opt, n_fft > win_length ??!
    spectrogram = torch.stft(a_tensor, STFT_N, STFT_HOP, STFT_WINDOW.size()[0], STFT_WINDOW,
            center=True, pad_mode='reflect', normalized=STFT_NORMALIZED, onesided=True)
    log_spectrogram = torch.log(spectrogram[:,:200,0] ** 2 + 1)     # Format size and set +1 (see noise level) to eliminate log(0)
    # normalize
    return log_spectrogram


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
                'output path': PATH_TO_VISUAL / ("{}_{}".format(yt_id, match) + ".jpg" if length == 0 else ".mp4"),
            },
            'audio': {
                #'audio codec': "-codec:a {}".format(A_CODEC),
                #'layout':
                # 'birate': '-b {}',
                'frequency': "-ar {}".format(A_FREQ),
                'channels': "-ac {}".format(A_CHANNELS),
                'output path': PATH_TO_AUDIO / ("{}_{}".format(yt_id, match) + ".wav"),
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